# -*- coding: utf-8 -*-

import os, sys
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from retrieval_model.passage_retrieval_enlarged import Retriever

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
from sklearn.metrics import f1_score
import json
import jsonlines

from tqdm import tqdm
import threading

import csv

from scipy.stats import spearmanr, chisquare
import argparse

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def accuracy(preds, labels):
    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1

    return 100 * (match_count / len(preds))


def accuracy_inst(pred, label):
    match_count = 0
    if pred == label:
        match_count = 1
    else:
        print('None equal', pred, label)
    return 100 * (match_count / 1)

def micro_F1(preds, labels):
    # Actually it is equal to the accuracy in the single-label non OOD setting
    result = f1_score(labels, preds, average='micro')
    return 100 * result

def macro_F1(preds, labels):
    # This differs from the accuracy even in the 2-class setting
    result = f1_score(labels, preds, average='macro')
    return 100 * result


def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0


def postprocess_answer_option_conditioned(answer):
    identity_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]", "[No Retrieval]", "[Retrieval]", "[Irrelevant]", "[Relevant]", "<paragraph>", "</paragraph>", "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]
    for token in identity_tokens:
        answer = answer.replace(token, "")

    if "</s>" in answer:
        answer = answer.replace("</s>", "")
    if "\n" in answer:
        answer = answer.replace("\n", "")

    if "<|endoftext|>" in answer:
        answer = answer.replace("<|endoftext|>", "")

    return answer


def find_identity_tokens(tokenizer, use_grounding=False, use_utility=False):
    retrieval_identity_tokens = ["[No Retrieval]", "[Retrieval]", "[Continue to Use Evidence]"]
    relativeness_identity_tokens = ["[Irrelevant]", "[Relevant]"]
    ground_identity_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    utility_identity_tokens = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_identity_tokens}
    rel_tokens = {}
    for token in relativeness_identity_tokens:
        # ["[Irrelevant]", "[Relevant]"]:
        rel_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    grd_tokens = None
    if use_grounding is True:
        grd_tokens = {}
        for token in ground_identity_tokens:
            grd_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    ut_tokens = None
    if use_utility is True:
        ut_tokens = {}
        for token in utility_identity_tokens:
            ut_tokens[token] = tokenizer.convert_tokens_to_ids(token)

    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def call_model_rerank_w_scores_batch(prompt, evidences, model, max_new_tokens=15,
                                     ret_tokens=None, rel_tokens=None, grd_tokens=None, ut_tokens=None,
                                     use_seqscore=False, threshold=0.5,
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=True, tokenizer=None):
    # tokenizer is for test only
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate([prompt], sampling_params)
        pred_token_ids = preds[0].outputs[0].token_ids
        pred_text = preds[0].outputs[0].text
        pred_log_probs = preds[0].outputs[0].logprobs
        results["no_retrieval"] = pred_text

    # save relevance token scores
    if mode == "always_retrieve":
        do_retrieve = True

    elif mode == "no_retrieval":
        do_retrieve = False
    else:
        if threshold is not None:
            score_dict = {}
            for tok, id in ret_tokens.items():
                if id not in pred_log_probs[0]:
                    score_dict[tok] = -100
                prob = pred_log_probs[0][id]
                score_dict[tok] = float(prob.logprob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred

    # Use beam retrieve. But only retrieve once, retrieve d < K=10 documents, so there is only one retrieval_token.
    # But there might be multiple relevance_tokens or grounding_tokens among the d documents.
    # See run_long_form_static_enlarged.py beam_width=2
    if do_retrieve is True:
        # gen all preds
        evidence_augmented_inputs = [prompt + "[Retrieval]<paragraph>{0}\n{1}</paragraph>".format(
            para["title"], para["text"]) for para in evidences]
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=5000)
        preds = model.generate(evidence_augmented_inputs, sampling_params)

        relevance_score_dict = {}
        grd_score_dict = {}
        ut_score_dict = {}
        overall_scores = {}
        for p_idx, pred in enumerate(preds):
            pred_token_ids = pred.outputs[0].token_ids
            pred_text = pred.outputs[0].text
            pred_log_probs = pred.outputs[0].logprobs
            seq_score = pred.outputs[0].cumulative_logprob / \
                max(len(pred.outputs[0].token_ids), 1)

            relevance_score_dict.setdefault(p_idx, {})
            grd_score_dict.setdefault(p_idx, {})
            ut_score_dict.setdefault(p_idx, {})
            # Compute reward scores
            # id is the token id, tok is the token string // not sequence
            for tok, id in rel_tokens.items():
                prob = pred_log_probs[0][id] if id in pred_log_probs[0] else -100
                relevance_score_dict[p_idx][tok] = np.exp(float(prob.logprob))

            if grd_tokens is not None:
                groundness_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(grd_tokens.values()):
                        groundness_token_appear_indices.append(tok_idx)
                        break
                if len(groundness_token_appear_indices) > 0:
                    # decoded_text = tokenizer.decode(pred_token_ids)
                    # Outputs [Relevant]NON-FACTUAL[Fully supported][Utility:5]</s> Alright, so pred.outputs[0].text automatically filter out the controller tokens.
                    # print(f'========>--------> len(groundness_token_appear_indices) > 0, {pred_text}, $$$$, {decoded_text}, |||||, {pred_token_ids}, ||---------||============')
                    idx = groundness_token_appear_indices[0]
                    for token, token_id in grd_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        grd_score_dict[p_idx][token] = np.exp(float(prob.logprob))

            if ut_tokens is not None:
                utility_token_appear_indices = []
                for tok_idx, tok in enumerate(pred_token_ids):
                    if tok in list(ut_tokens.values()):
                        utility_token_appear_indices.append(tok_idx)
                if len(utility_token_appear_indices) > 0:
                    idx = utility_token_appear_indices[0]
                    for token, token_id in ut_tokens.items():
                        prob = pred_log_probs[idx][token_id] if token_id in pred_log_probs[idx] else -100
                        ut_score_dict[p_idx][token] = np.exp(float(prob.logprob))

            relevance_score = relevance_score_dict[p_idx]["[Relevant]"] / (
                np.sum(list(relevance_score_dict[p_idx].values())))

            if len(grd_score_dict[p_idx]) == 3:
                gt_sum = np.sum(list(grd_score_dict[p_idx].values()))
                ground_score = (grd_score_dict[p_idx]["[Fully supported]"] / gt_sum) + 0.5 * (
                    grd_score_dict[p_idx]["[Partially supported]"] / gt_sum)
                # print(f'grd_score_dict[{p_idx}] is 3, the text is: {pred_text}<====================!')
                decoded_text = tokenizer.decode(pred_token_ids)
                print(f'grd_score_dict[{p_idx}] is 3, the text is: {pred_text}<===================={decoded_text}<====!!!')
            else:
                # sys.exit('grd_score_dict[p_idx] is not 3<====================!!!')
                decoded_text = tokenizer.decode(pred_token_ids)
                print(f'grd_score_dict[{p_idx}] is not 3, the text is: {pred_text}<===================={decoded_text}<====!!!')
                ground_score = 0.0

            # if ut_tokens is None then utility_score = 0
            if len(ut_score_dict[p_idx]) == 5:
                ut_sum = np.sum(list(ut_score_dict[p_idx].values()))
                ut_scores = [-1, -0.5, 0, 0.5, 1]
                utility_score = np.sum(
                    [ut_scores[i] * (ut_score_dict[p_idx]["[Utility:{}]".format(i+1)] / ut_sum) for i in range(len(ut_scores))])
            else:
                utility_score = 0.0

            if use_seqscore is True:
                final_score = np.exp(seq_score) + w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score
            else:
                final_score = w_rel * relevance_score + \
                    w_sup * ground_score + w_use * utility_score

            overall_scores[p_idx] = {"final_score": final_score,
                                     "relevance_score": relevance_score,
                                     "ground_score": ground_score,
                                     "utility_score": utility_score,
                                     "relevance_score_dict": relevance_score_dict,
                                     "grd_score_dict": grd_score_dict,
                                     "ut_score_dict": utility_score,
                                     "truthfulness_score": final_score / (w_rel + w_sup)}
            results["retrieval_{}".format(p_idx)] = {
                "pred": pred_text, "score": final_score, "ctx": evidences[p_idx], "truthfulness_score": final_score / (w_rel + w_sup)}

    else:
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens)
        prompt += "[No Retrieval]"
        preds = model.generate([prompt], sampling_params)

        pred = preds[0].outputs[0].text

    # Aggregating answers
    if len(results) == 1:
        # result = {p_idx: ...}
        # only one pred, so can use the one in iteration of preds, which is the only one
        postprocessed_pred = postprocess_answer_option_conditioned(pred)
        if do_retrieve is True:
            best_overall_score = overall_scores[0]
            best_aggregated_answer_truthfulnessscore_mean = best_overall_score['truthfulness_score']
        else:
            best_aggregated_answer_truthfulnessscore_mean = 0.5
        best_overall_score = None
        overall_scores = None
        return postprocessed_pred, results, do_retrieve, best_overall_score, overall_scores, best_aggregated_answer_truthfulnessscore_mean
    else:
        # when entering this branch, do_retrieve is always True
        answer2score = {}
        answer2truthfulness_score = {}
        answer2appearance_count = {}
        # closed choice
        if closed is True:
            for key, result in results.items():
                # print('------------>>>>> key:', key)
                # ------------>>>>> key: no_retrieval
                # ------------>>>>> key: retrieval_0
                # ------------>>>>> key: retrieval_1
                if key == "no_retrieval":
                    # Here all the results are generated by using retrieval so no_retrieval is omitted.
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                if answer2score[answer] != 0:
                    print('================ Different evidences generated the same answer more than once! ================')
                answer2score[answer] += score
                answer2appearance_count.setdefault(answer, 0)
                answer2appearance_count[answer] += 1
                answer2truthfulness_score.setdefault(answer, 0)
                answer2truthfulness_score[answer] += result["truthfulness_score"]
            answer2score_mean = {key: item/answer2appearance_count[key] for key, item in answer2score.items()}
            sorted_answers = sorted(
                answer2score_mean.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
            best_aggregated_answer_truthfulnessscore_mean = answer2truthfulness_score[best_option] / answer2appearance_count[best_option]

            score_path2score = {key: item["final_score"] for key, item in overall_scores.items()}
            sorted_score_path = sorted(score_path2score.items(), key=lambda x: x[1], reverse=True)
            print('========', sorted_answers[0][1], sorted_score_path[0][1], '<============')
            # assert sorted_answers[0][1] == sorted_score_path[0][1]
            best_overall_score =  overall_scores[sorted_score_path[0][0]]
        else:
            print('Open choice is implemented, but shouldnt be entered in this scenario!')
            sys.exit('!!!!!!!!!!Open choice is implemented, but shouldnt be entered in this scenario!')
            path2score = {key: item["score"] for key,
                          item in results.items() if key != "no_retrieval"}
            sorted_path2score = sorted(path2score.items(),
                               key=lambda x: x[1], reverse=True)
            best_path = sorted_path2score[0][0]
            best_score = sorted_path2score[0][1]
            best_option = results[best_path]["pred"]
            # Here, no aggregation
            best_aggregated_answer_truthfulnessscore_mean = results[best_path]["truthfulness_score"]

            score_path2score = {key: item["final_score"] for key, item in overall_scores.items()}
            sorted_score_path = sorted(score_path2score.items(), key=lambda x: x[1], reverse=True)
            assert results[best_path]["score"] == sorted_score_path[0][1]
            assert best_score == sorted_score_path[0][1]
            best_overall_score = overall_scores[sorted_score_path[0][0]]

        best_overall_score = None
        overall_scores = None
        return best_option, results, do_retrieve, best_overall_score, overall_scores, best_aggregated_answer_truthfulnessscore_mean


def retrieve_input_data_multithread(retriever, dataset_sublist, retults):
    # new_data_sublist = []
    for i, item in enumerate(dataset_sublist):
        retrieved_documents = retriever.search_document_demo(item["question"], 10)
        retults.append(retrieved_documents)
    # return new_data_sublist


def preprocess_input_data(dataset):
    new_data = []
    instruction = "Is the following statement correct or not? Say FACTUAL if it's correct; otherwise say NON-FACTUAL."

    for i, item in tqdm(enumerate(dataset)):
        item["question"] = item['response']
        prompt = instruction + "\n\n## Input:\n\n" + \
            item["question"] if instruction is not None else item["question"]
        item["instruction"] = prompt

    retriever = Retriever({})
    # K = 10
    retriever.setup_retriever_demo("facebook/contriever-msmarco", "./retrieval_lm/psgs_w100.tsv", "./retrieval_lm/wikipedia_embeddings/*",  n_docs=10, save_or_load_index=True)
    print('retriever-------------------done!')
    lst_docidx2evidences = []
    lst_threads = []
    # 40 threads
    N_threads = 40
    lst_threadidx2results = [list() for _ in range(N_threads)]
    seconds = 1.7 * 10 * len(dataset)
    waiting_time = seconds / N_threads
    print(f'total execution time: {seconds}, please wait for {waiting_time}, the number of threads is {N_threads}')
    print(f'retrieved_documents multithread thread_count={N_threads}-------------------START!')
    for i in range(N_threads):
        if i == N_threads - 1:
            dataset_sublist = dataset[i * len(dataset) // N_threads:]
        else:
            dataset_sublist = dataset[i * len(dataset) // N_threads: (i + 1) * len(dataset) // N_threads]
        thread = threading.Thread(target=retrieve_input_data_multithread, args=(retriever, dataset_sublist, lst_threadidx2results[i],))
        lst_threads.append(thread)
        thread.start()
    for i, thread in enumerate(lst_threads):
        thread.join()
        lst_docidx2evidences.extend(lst_threadidx2results[i])
    print('retrieved_documents multithread------------------------------done!')
    print('len(lst_docidx2evidences):', len(lst_docidx2evidences))

    for i, item in tqdm(enumerate(dataset)):
        item["ctxs"] = lst_docidx2evidences[i]
        new_data.append(item)

    return new_data

def process_a_dataset(output_file='./output_result', output_acc_truthfulness_file='./output_acc_truthfulness', max_new_tokens=15, ndocs=3, world_size=4, dtype='half', threshold=0.5, use_seqscore=False, use_groundness=True, use_utility=False, w_rel=1, w_sup=1, w_use=1, mode='adaptive_retrieval', metric='accuracy'):
    # preprocess_data
    input_data = load_jsonlines('./FactCHD/raw_test_200.json')
    # input_data = load_jsonlines('./FactCHD/raw_test_2k.json')
    # input_data = load_jsonlines('./FactCHD/raw_test.json')
    print('input_data----------------------------------loaded')
    input_data = preprocess_input_data(input_data)
    tokenizer = AutoTokenizer.from_pretrained('selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_elandi/.cache/huggingface/hub", padding_side="left")
    model = LLM(model='selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_elandi/.cache/huggingface/hub", dtype=dtype, tensor_parallel_size=world_size,max_logprobs=5000)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = find_identity_tokens(
        tokenizer, use_grounding=use_groundness, use_utility=use_utility)
    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=threshold, use_seqscore=use_seqscore,
                                                w_rel=w_rel, w_sup=w_sup, w_use=w_use, mode=mode, closed=True, tokenizer=tokenizer)
    # closed=True means that the model is trained in a closed-book setting, where the model is trained to select the best answer from a fixed set of answer options.

    preds = []
    prompts = []
    golds = []
    metric_results = []

    metric_results_batched = []
    metric_results_interim_batched = []
    lst_best_aggregated_answer_truthfulnessscore_means_batched = []
    lst_best_aggregated_answer_truthfulnessscore_means_interim_batched = []
    batch_size_for_metric_results = 200
    metric_results_among_usingretrieval_batched = []
    metric_results_batched_among_usingretrieval_interim_batched = []
    lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly_batched = []
    lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly_interim_batched = []
    
    scores = []
    all_results = []
    lst_best_aggregated_answer_truthfulnessscore_means = []
    lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly = []
    do_retrieves = []
    metric_results_among_usingretrieval = []
    count_retrieve = 0
    count = 0
    prompt_format_string = "### Instruction:\n{instruction}\n\n### Response:\n"
    for i, row in tqdm(enumerate(input_data)):
        results = {}
        # instruction will be added to the prompt
        prompt = prompt_format_string.format_map(row)
        # print('row[ctxs]:', row['ctxs'])
        evidences = row['ctxs'][:ndocs]
        pred, results, do_retrieve, best_overall_score, overall_scores, best_aggregated_answer_truthfulnessscore_mean = generate(
            prompt, evidences, max_new_tokens=max_new_tokens,)
        if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
        prompts.append(prompt)
        preds.append(pred)
        all_results.append(results)
        do_retrieves.append(do_retrieve)
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if metric == "accuracy":
            # metric_result = accuracy(pred, row["output"])
            metric_result = accuracy_inst(pred, row["label"])
        else:
            raise NotImplementedError
        if do_retrieve is True:
            count_retrieve += 1
            lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly.append(best_aggregated_answer_truthfulnessscore_mean)
            lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly_interim_batched.append(best_aggregated_answer_truthfulnessscore_mean)
            metric_results_among_usingretrieval.append(metric_result)
            metric_results_batched_among_usingretrieval_interim_batched.append(metric_result)
        count += 1
        metric_results.append(metric_result)
        metric_results_interim_batched.append(metric_result)
        lst_best_aggregated_answer_truthfulnessscore_means.append(best_aggregated_answer_truthfulnessscore_mean)
        lst_best_aggregated_answer_truthfulnessscore_means_interim_batched.append(best_aggregated_answer_truthfulnessscore_mean)
        if i % 10 == 0:
            print("average: metric_result={}, truthfulness_score={}, metric_result_among_usingretrieval={}, truthfulness_score_among_usingretrieval={}".format(np.mean(metric_results), np.mean(lst_best_aggregated_answer_truthfulnessscore_means), np.mean(metric_results_among_usingretrieval), np.mean(lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly)))
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  metric, "metric_mean": np.mean(metric_results), "scores": scores, "truthfulness_score_mean": np.mean(lst_best_aggregated_answer_truthfulnessscore_means), "truthfulness_score": lst_best_aggregated_answer_truthfulnessscore_means}
            # with open(output_file + "_tmp", "w") as outfile:
            #     json.dump(final_results, outfile)
        if i % batch_size_for_metric_results == 0:
            metric_results_batched.append(np.mean(metric_results_interim_batched))
            metric_results_interim_batched = []
            lst_best_aggregated_answer_truthfulnessscore_means_batched.append(np.mean(lst_best_aggregated_answer_truthfulnessscore_means_interim_batched))
            lst_best_aggregated_answer_truthfulnessscore_means_interim_batched = []
            metric_results_among_usingretrieval_batched.append(np.mean(metric_results_batched_among_usingretrieval_interim_batched))
            metric_results_batched_among_usingretrieval_interim_batched = []
            lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly_batched.append(np.mean(lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly_interim_batched))
            lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly_interim_batched = []

    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  metric, "metric_mean": np.mean(metric_results), "scores": scores, "truthfulness_score_mean": np.mean(lst_best_aggregated_answer_truthfulnessscore_means), "truthfulness_score": lst_best_aggregated_answer_truthfulnessscore_means, "do_retrieves": do_retrieves}
    
    print("Final result: average: metric_result={0}, truthfulness_score={1}, metric_result_among_usingretrieval={2}, truthfulness_score_among_usingretrieval={3}".format(np.mean(metric_results), np.mean(lst_best_aggregated_answer_truthfulnessscore_means), np.mean(metric_results_among_usingretrieval), np.mean(lst_best_aggregated_answer_truthfulnessscore_means_retrieveTrueOnly)))
    # print("Retrieval Frequencies: {0}".format(count_retrieve / len(final_results)))
    print("Retrieval Frequencies: {0}".format(count_retrieve / len(input_data)))

    outfile_acc_tru = open(output_acc_truthfulness_file, "w")
    csv_writer = csv.writer(outfile_acc_tru, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["accuracy", "truthfulness_score"])
    for i in range(len(metric_results_batched)):
        csv_writer.writerow([metric_results_batched[i], lst_best_aggregated_answer_truthfulnessscore_means_batched[i]])
    outfile_acc_tru.close()
    print('Done!')

def process_an_assertion(retriever=None, an_assertion=None, tokenizer=None, llm_model=None, identity_tokens_tuple=None, max_new_tokens=15, ndocs=3, world_size=4, dtype='half', threshold=0.5, use_seqscore=False, use_groundness=True, use_utility=False, w_rel=1, w_sup=1, w_use=1, mode='adaptive_retrieval', metric='accuracy'):
    if retriever is None:
        retriever = Retriever({})
        # K = 10
        retriever.setup_retriever_demo("facebook/contriever-msmarco", "./retrieval_lm/psgs_w100.tsv", "./retrieval_lm/wikipedia_embeddings/*",  n_docs=40, save_or_load_index=True)
        print('retriever-------------------done!')
    
    # 'The Gaussian distribution format is N(0, I).'
    if an_assertion is None:
        an_assertion = input('Please input an assertion: ')
        instruction = "Is the following statement correct or not? Say FACTUAL if it's correct; otherwise say NON-FACTUAL."
        prompt = instruction + "\n\n## Input:\n\n" + an_assertion
    else:
        instruction = "Is the following statement correct or not? Say FACTUAL if it's correct; otherwise say NON-FACTUAL."
        prompt = instruction + "\n\n## Input:\n\n" + an_assertion
    retrieved_documents = retriever.search_document_demo(an_assertion, 10)

    # tokenizer = AutoTokenizer.from_pretrained('selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_elandi/.cache/huggingface/hub", padding_side="left")
    model = llm_model
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = identity_tokens_tuple

    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=threshold, use_seqscore=use_seqscore,
                                                w_rel=w_rel, w_sup=w_sup, w_use=w_use, mode=mode, closed=True, tokenizer=tokenizer)
    # closed=True means that the model is trained in a closed-book setting, where the model is trained to select the best answer from a fixed set of answer options.
    pred, results, do_retrieve, best_overall_score, overall_scores, best_aggregated_answer_truthfulnessscore_mean = generate(
            prompt, retrieved_documents, max_new_tokens=max_new_tokens,)
    if type(pred) is str and pred[0] == "#" or pred[0] == ":":
            pred = pred[1:]
    print('pred:', pred)
    print('do_retrieve:', do_retrieve)
    print('aggregated_truthfulnessscore_mean:', best_aggregated_answer_truthfulnessscore_mean)
    for key, result in results.items():
        # print('------------>>>>> key:', key)
        # ------------>>>>> key: no_retrieval
        # ------------>>>>> key: retrieval_0
        # ------------>>>>> key: retrieval_1
        if key == "no_retrieval":
            # Here all the results are generated by using retrieval so no_retrieval is omitted.
            continue
        if pred == result['pred']:
            link = 'https://en.wikipedia.org/wiki/' + result['ctx']['title'].replace(' ', '_')
            print('key: {}, pred: {}, evidence: {}, wikipedia-page: {} , truthfulness_score_single_evidence: {}'.format(key, result['pred'], result['ctx'], link, result['truthfulness_score']))
        else:
            continue
        # print('key: {}, pred: {}, evidence: {}, truthfulness_score_single_evidence: {}\n'.format(key, result['pred'], result['ctx'], result['truthfulness_score']))


def demo_interactive_process_assertion(max_new_tokens=15, ndocs=10, world_size=4, dtype='half', threshold=0.5, use_seqscore=False, use_groundness=True, use_utility=False, w_rel=1, w_sup=1, w_use=1, mode='adaptive_retrieval', metric='accuracy'):
    retriever = Retriever({})
    # K = 10
    retriever.setup_retriever_demo("facebook/contriever-msmarco", "./retrieval_lm/psgs_w100.tsv", "./retrieval_lm/wikipedia_embeddings/*",  n_docs=40, save_or_load_index=True)
    print('retriever-------------------done!')

    tokenizer = AutoTokenizer.from_pretrained('selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_elandi/.cache/huggingface/hub", padding_side="left")
    # model = LLM(model='selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_llmcache/vllm_cache", dtype=dtype, tensor_parallel_size=world_size,max_logprobs=5000)
    model = LLM(model='selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_elandi/.cache/huggingface/hub", dtype=dtype, tensor_parallel_size=world_size,max_logprobs=5000)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = find_identity_tokens(
        tokenizer, use_grounding=use_groundness, use_utility=use_utility)
    identity_tokens_tuple = (ret_tokens, rel_tokens, grd_tokens, ut_tokens)
    print('tokenizer, model, identity_tokens_tuple-------------------done!')

    while True:
        process_an_assertion(retriever=retriever, tokenizer=tokenizer, llm_model=model, identity_tokens_tuple=identity_tokens_tuple, max_new_tokens=max_new_tokens, ndocs=ndocs, world_size=world_size, dtype=dtype, threshold=threshold, use_seqscore=use_seqscore, use_groundness=use_groundness, use_utility=use_utility, w_rel=w_rel, w_sup=w_sup, w_use=w_use, mode=mode, metric=metric)

def main(**argv):
    demo_interactive_process_assertion(
        max_new_tokens=argv['max_new_tokens'],
        ndocs=argv['ndocs'],
        world_size=argv['world_size'],
        dtype=argv['dtype'],
        threshold=argv['threshold'],
        w_rel=argv['w_rel'],
        w_sup=argv['w_sup'],
        mode=argv['mode']
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', default='15', type=int, required=False, help='max_tokens param for SamplingParams of vllm, default=15, since the generated text is FACTUAL or NON-FACTUAL here.')
    parser.add_argument('--ndocs', default='10', type=int, required=False, help='number of documents to probe in the beam search, default=10')
    parser.add_argument('--world_size', default='1', type=int, required=False, help='number of GPUs to run the model, depending on and must less than os.environ[\'CUDA_VISIBLE_DEVICES\'], default=1')
    parser.add_argument('--dtype', default='half', type=str, required=False, help='dtype param for LLM, default=half')
    parser.add_argument('--threshold', default=0.5, type=float, required=False, help='threshold for retrieval, default=0.5')
    parser.add_argument('--w_rel', default=1, type=int, required=False, help='weight of relevance score, default=1')
    parser.add_argument('--w_sup', default=1, type=int, required=False, help='weight of supportiveness score, default=1')
    parser.add_argument('--mode', default='adaptive_retrieval', type=str, required=False, help='Control if using retrieval or not, can be [always_retrieve|adaptive_retrieval|no_retrieval], default=adaptive_retrieval')

    my_args = '--world_size 1'
    args = parser.parse_args(my_args.split(' '))
    # args = parser.parse_args() # if you want to use sys.argv that is the terminal argvs
    # print(args) # It's a namespace
    # namespace to dict
    py_argv = vars(args)
    main(**py_argv)

    # process_a_dataset(world_size=1)

    # # 3 Testing cases
    # # Hyperparameter tuning is a critical step in the machine learning pipeline, especially for complex models like Random Forests. It involves adjusting the parameters that govern the learning process, which are not learned from the data but set prior to training. The performance of a Random Forest model can significantly vary based on the choice of hyperparameters such as the number of trees, maximum depth, and minimum samples per leaf. Proper tuning can lead to improved accuracy, reduced overfitting, and enhanced generalization to unseen data. Techniques for hyperparameter optimization include Grid Search, which exhaustively searches through a specified subset of hyperparameters, and Random Search, which samples a wide range of values. More advanced methods like Bayesian Optimization can also be employed, which builds a probabilistic model of the function mapping hyperparameters to model performance and uses it to select the most promising hyperparameters to evaluate next. Cross-validation is often used in conjunction with these techniques to ensure that the model's performance is robust and not merely a result of overfitting to the training data. Ultimately, effective hyperparameter tuning is essential for maximizing the predictive performance of Random Forests and ensuring that the model is well-suited for the specific characteristics of the dataset at hand.

    # # Markov Random Fields (MRFs) are a powerful framework within probabilistic graphical models that enable the representation of complex dependencies among random variables. In the context of image processing and spatial statistics, MRFs are particularly useful for modeling spatial relationships and capturing local dependencies in data. The potential functions in MRFs define the interactions between neighboring pixels or variables, allowing for the encoding of prior knowledge about the structure of the data. These functions are crucial for tasks such as image segmentation and object recognition, where the goal is to classify each pixel based on its context within the image. Conditional independence is a key property of MRFs that simplifies the computation of joint distributions. It allows for the decomposition of complex problems into more manageable subproblems, facilitating efficient inference methods such as Belief Propagation and Gibbs Sampling. These methods leverage the local structure of MRFs to compute marginal distributions and maximum a posteriori (MAP) estimates effectively. In practical applications, such as medical image analysis and social network analysis, MRFs provide a robust framework for modeling dependencies and making predictions based on observed data. The theoretical foundations of MRFs, combined with their graphical representation, enable researchers and practitioners to develop sophisticated models that can capture the intricacies of real-world phenomena, ultimately leading to improved decision-making and insights across various domains.

    # # Federated learning (FL) presents a transformative approach in healthcare by enabling collaborative model training across decentralized data sources while preserving data privacy. This is particularly crucial in sensitive domains like healthcare, where patient data is subject to stringent regulatory compliance. By allowing models to be trained locally on devices or institutions without sharing raw data, FL mitigates privacy concerns and adheres to regulations such as HIPAA. In the context of predictive analytics, FL can significantly enhance the accuracy of models used in medical imaging and genomic data analysis. For instance, by aggregating insights from diverse datasets, FL can improve the robustness of predictive models, leading to better diagnostic tools and personalized treatment plans. However, the implementation of FL in healthcare is not without challenges. Data heterogeneity, which refers to the variability in data distributions across different institutions, can complicate model training and performance. Additionally, communication efficiency is a critical factor, as frequent model updates can lead to increased bandwidth usage and latency. To address these challenges, techniques such as model aggregation and privacy-preserving methods like differential privacy and homomorphic encryption can be employed. These techniques ensure that while models learn from diverse data sources, the individual data points remain secure. Ultimately, the successful integration of federated learning in healthcare hinges on balancing the need for advanced analytics with the imperatives of data privacy and regulatory compliance, paving the way for innovative and ethical AI practices in the field.
