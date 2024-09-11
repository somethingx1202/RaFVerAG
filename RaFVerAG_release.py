# -*- coding: utf-8 -*-

import os, sys
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

# from CLASS_SR_RAG import SR_RAG
# # from retrieval_lm.passage_retrieval import Retriever
from retrieval_lm.passage_retrieval_enlarged import Retriever
# # from vllm import LLM, SamplingParams

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import numpy as np
from sklearn.metrics import f1_score
import json
import jsonlines

from tqdm import tqdm
import threading

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
    # else:
    #     print('None equal', pred, label)
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
    ground_identity_tokens = ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"]
    utility_identity_tokens = ["[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]"]

    ret_tokens = {token: tokenizer.convert_tokens_to_ids(
        token) for token in retrieval_identity_tokens}
    rel_tokens = {}
    for token in ["[Irrelevant]", "[Relevant]"]:
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
                                     w_rel=1.0, w_sup=1.0, w_use=0.5, mode="adaptive_retrieval", closed=True):
    results = {}
    if mode != "always_retrieve":
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, logprobs=32016)
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
                # print(f'prob: {prob}<====================')
                # print('prob:', prob.logprob, '<====================')
                # prob: Logprob(logprob=-7.700498104095459, rank=33, decoded_token='')
                score_dict[tok] = float(prob.logprob)
            do_retrieve = score_dict["[Retrieval]"] / (
                score_dict["[Retrieval]"] + score_dict["[No Retrieval]"]) > threshold
        else:
            do_retrieve = "[Retrieval]" in pred

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
            else:
                ground_score = 0.0

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
        # best_overall_score = overall_scores.items()[0][1]
        # print('do_retrieve:', do_retrieve) # False can happen
        if do_retrieve is True:
            sys.exit('do_retrieve is True, but only one result is generated!')
            best_overall_score = overall_scores[0]
            best_aggregated_answer_truthfulnessscore_mean = best_overall_score['truthfulness_score']
        else:
            best_aggregated_answer_truthfulnessscore_mean = 0.5
        best_overall_score = None
        overall_scores = None
        return postprocessed_pred, results, do_retrieve, best_overall_score, overall_scores, best_aggregated_answer_truthfulnessscore_mean
    else:
        answer2score = {}
        answer2truthfulness_score = {}
        answer2appearance_count = {}
        if closed is True:
            # 这边其实可以用p_idx在overall_scores里面遍历，然后key用"retrieval_{}".format(p_idx) 来转换成results里面的key
            for key, result in results.items():
                if key == "no_retrieval":
                    continue
                answer = postprocess_answer_option_conditioned(result["pred"])
                score = result["score"]
                answer2score.setdefault(answer, 0)
                # if answer2score[answer] != 0:
                #     print('================ Different evidences generated the same answer more than once! ================')
                # ranking score = 0.9 relevance_score + 0.1 supportiveness
                # truthfulness_score = 0.5 relevan
                # indenpendent.
                answer2score[answer] += score
                answer2appearance_count.setdefault(answer, 0)
                answer2appearance_count[answer] += 1
                answer2truthfulness_score.setdefault(answer, 0)
                answer2truthfulness_score[answer] += result["truthfulness_score"]
            sorted_answers = sorted(
                answer2score.items(), key=lambda x: x[1], reverse=True)
            best_option = sorted_answers[0][0]
            best_aggregated_answer_truthfulnessscore_mean = answer2truthfulness_score[best_option] / answer2appearance_count[best_option]

            # path2score = {key: item["score"] for key,
            #               item in results.items() if key != "no_retrieval"}
            # best_path = sorted(path2score.items(),
            #                    key=lambda x: x[1], reverse=True)[0][0]
            score_path2score = {key: item["final_score"] for key, item in overall_scores.items()}
            sorted_score_path = sorted(score_path2score.items(), key=lambda x: x[1], reverse=True)
            print('========', sorted_answers[0][1], sorted_score_path[0][1], '<============')
            # assert sorted_answers[0][1] == sorted_score_path[0][1]
            best_overall_score =  overall_scores[sorted_score_path[0][0]]
        else:
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
    retriever.setup_retriever_demo("facebook/contriever-msmarco", "./retrieval_lm/psgs_w100.tsv", "./retrieval_lm/wikipedia_embeddings/*",  n_docs=10, save_or_load_index=True)

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
        # item["question"] = item['response']
        # prompt = instruction + "\n\n## Input:\n\n" + \
        #     item["question"] if instruction is not None else item["question"]
        # item["instruction"] = prompt
        # # retrieved_documents = retriever.search_document_demo(item["question"], 10)
        # # item["evidences"] = retrieved_documents
        item["ctxs"] = lst_docidx2evidences[i]
        new_data.append(item)

    return new_data


# qa_f1_score: F1 score for one instance (actually a sequence of answer). # and for each answer, it's a intersection over two unions of word frequency of a pair of answers.
def process_a_dataset(output_file='./output_result', max_new_tokens=15, ndocs=3, world_size=4, dtype='half', threshold=0.5, use_seqscore=False, use_groundness=True, use_utility=False, w_rel=1, w_sup=1, w_use=1, mode='adaptive_retrieval', metric='accuracy'):
    # preprocess_data
    input_data = load_jsonlines('./FactCHD/raw_test_200.json')
    print('input_data----------------------------------loaded')
    input_data = preprocess_input_data(input_data)
    tokenizer = AutoTokenizer.from_pretrained('selfrag/selfrag_llama2_13b', padding_side="left")
    model = LLM(model='selfrag/selfrag_llama2_13b', download_dir="/scratch/prj/inf_llmcache/vllm_cache", dtype=dtype, tensor_parallel_size=world_size,max_logprobs=32016)
    ret_tokens, rel_tokens, grd_tokens, ut_tokens = find_identity_tokens(
        tokenizer, use_grounding=use_groundness, use_utility=use_utility)
    # if ut_tokens is None:
    #     sys.exit('ut_tokens is indeed None when use_utility is False<====================') # This is correct
    # else:
    #     sys.exit('ut_tokens is NOT None!!!!!!<=====================')
    def generate(prompt, evidences, max_new_tokens):
        return call_model_rerank_w_scores_batch(prompt, evidences=evidences, model=model, max_new_tokens=max_new_tokens,
                                                rel_tokens=rel_tokens, ret_tokens=ret_tokens, grd_tokens=grd_tokens, ut_tokens=ut_tokens,
                                                threshold=threshold, use_seqscore=use_seqscore,
                                                w_rel=w_rel, w_sup=w_sup, w_use=w_use, mode=mode, closed=True)
    # closed=True means that the model is trained in a closed-book setting, where the model is trained to select the best answer from a fixed set of answer options.

    preds = []
    prompts = []
    golds = []
    metric_results = []
    scores = []
    all_results = []
    lst_best_aggregated_answer_truthfulnessscore_means = []
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
        if do_retrieve is True:
            count += 1
        if "answers" not in row and "answer" in row:
            row["answers"] = [row["answer"]] if type(
                row["answer"]) is str else row["answer"]
        if metric == "accuracy":
            # metric_result = accuracy(pred, row["output"])
            metric_result = accuracy_inst(pred, row["label"])
        else:
            raise NotImplementedError

        metric_results.append(metric_result)
        lst_best_aggregated_answer_truthfulnessscore_means.append(best_aggregated_answer_truthfulnessscore_mean)
        if i % 10 == 0:
            print("average: metric_result={}, truthfulness_score={}".format(np.mean(metric_results), np.mean(lst_best_aggregated_answer_truthfulnessscore_means)))
            final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                             "golds": golds,  "metric":  metric, "metric_mean": np.mean(metric_results), "scores": scores, "truthfulness_score_mean": np.mean(lst_best_aggregated_answer_truthfulnessscore_means), "truthfulness_score": lst_best_aggregated_answer_truthfulnessscore_means}
            with open(output_file + "_tmp", "w") as outfile:
                json.dump(final_results, outfile)

    final_results = {"preds": preds, "prompts": prompts, "metric_results": metric_results, "all_results": all_results,
                     "golds": golds,  "metric":  metric, "metric_mean": np.mean(metric_results), "scores": scores, "truthfulness_score_mean": np.mean(lst_best_aggregated_answer_truthfulnessscore_means), "truthfulness_score": lst_best_aggregated_answer_truthfulnessscore_means}
    with open(output_file, "w") as outfile:
        json.dump(final_results, outfile)

    print("Final result: average: metric_result={0}, truthfulness_score={1}".format(np.mean(metric_results), np.mean(lst_best_aggregated_answer_truthfulnessscore_means)))
    # print("Retrieval Frequencies: {0}".format(count / len(final_results)))
    print("Retrieval Frequencies: {0}".format(count / len(input_data)))
    

if __name__ == '__main__':
    process_a_dataset(world_size=4)
    