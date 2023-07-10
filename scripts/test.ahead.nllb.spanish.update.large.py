import os
from  transformers import AutoTokenizer
from  transformers import AutoModelForSeq2SeqLM
import torch
import math
import string
import pickle
from wordfreq import zipf_frequency
punctuation=string.punctuation

import deepspeed
local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
device = torch.device("cuda", local_rank)

model_name_or_path="model/facebook/nllb-200-3.3B"

tokenizer = AutoTokenizer.from_pretrained(

    model_name_or_path, src_lang="spa_Latn"

)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"]).eval()
model=deepspeed.init_inference(model,
                                mp_size=world_size,
                                replace_method="auto")

def cal_freq_scores_spanish(substitutes):
    scores=[]
    for word1 in substitutes:
        tmp_score=zipf_frequency(word1, 'es')
        scores.append(tmp_score)

    return scores

ori_sentences=open("data/tsar/tsar2022_es_test_gold.tsv",encoding="utf-8").readlines()
from nltk.tokenize.toktok import ToktokTokenizer
spanish_tok = ToktokTokenizer()
qutos=["</s>","<pad>","<s>"]
def skip_words(word):
    word=word.replace("spa_Latn","")
    word=word.replace("<s>","")
    word=word.replace("<pad>","")
    word=word.replace("</s>","")
    return word.strip()
from nltk.stem.snowball import SnowballStemmer
spanish_ps=SnowballStemmer("spanish")


print("beding load word2vec")
from gensim.models.fasttext import load_facebook_model
word2vec_model_path = 'model/word2vec/cc.es.300.bin.gz'
word_vec_model = load_facebook_model(word2vec_model_path)
print("finishe loading word2vec model!!")
def cal_static_score(complex_word,candis):  
    if complex_word not in word_vec_model.wv.key_to_index.keys():
        return torch.zeros(len(candis)).tolist()
    candis_scores=[]
    for candi in candis:
        if candi not in word_vec_model.wv.key_to_index.keys():
            candis_scores.append(-1)
        else:
            candis_scores.append(word_vec_model.wv.similarity(complex_word, candi))

    return candis_scores



@torch.no_grad()
def give_real_scores_ahead(tokenizer,outputs,scores_with_suffix,scores_with_suffix_masks,suffix_tokens,prefix_len=None,prefix_str=None,max_ahead=1,flag=1):
    beam_size,max_len=outputs.size()
    scores_with_suffix=scores_with_suffix[:,:max_len]
    scores_with_suffix_masks=scores_with_suffix_masks[:,:max_len]

    first_index=prefix_len+2
    last_index=min(first_index+max_ahead,max_len)

    ahead_parts=outputs[:,1:]
    ahead_parts=ahead_parts.reshape(1,-1)[0].tolist()
    ahead_part_tokens=list(map(lambda x:tokenizer.convert_ids_to_tokens(x),ahead_parts))
    ahead_part_tokens_masks=list(map(lambda x:not x.startswith("▁") and x not in qutos,ahead_part_tokens))
    ahead_part_tokens_masks=torch.tensor(ahead_part_tokens_masks)
    ahead_part_tokens_masks=ahead_part_tokens_masks.reshape(beam_size,-1)
    scores_with_suffix[:,:-1][ahead_part_tokens_masks]=-math.inf
    scores_with_suffix[scores_with_suffix_masks]=-math.inf 
    for j in range(0,first_index):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)

    for j in range(last_index,max_len):
        scores_with_suffix[:,j]=torch.tensor(-math.inf)   

    flat_scores_with_suffix=scores_with_suffix.reshape(1,-1).squeeze(dim=0)
    sorted_scores,sorted_indices=torch.topk(flat_scores_with_suffix,k=beam_size*max_ahead)
    beam_idx=sorted_indices//max_len
    len_idx=(sorted_indices%max_len)
    
    if flag!=None:
        hope_len=len(spanish_tok.tokenize(prefix_str))+flag
    else:
        hope_len=-1

    hope_outputs=[]
    hope_outputs_scores=[]
    candis=[]

    for i in range(len(beam_idx)):
        if sorted_scores[i]==(-math.inf):
            continue

        tmp_str1=" ".join(tokenizer.convert_ids_to_tokens(outputs[beam_idx[i],:(len_idx[i]+1)])).replace(" ","").replace("▁"," ").strip()
        tmp_str1=skip_words(tmp_str1).strip()
        if len(spanish_tok.tokenize(tmp_str1))==hope_len:
            hope_outputs.append(outputs[beam_idx[i]])
            hope_outputs_scores.append(sorted_scores[i].tolist())
            candis.append(spanish_tok.tokenize(tmp_str1)[-1].strip())

        elif hope_len==-1:
            hope_outputs.append(outputs[beam_idx[i],:(len_idx[i]+1)])
            hope_outputs_scores.append(sorted_scores[i].tolist())
    return hope_outputs,hope_outputs_scores,candis





@torch.no_grad()
def lexicalSubstitute(model,tokenizer, sentence, complex_word,prefix,suffix,beam):
    sentence_tokens= tokenizer.encode(sentence, return_tensors='pt')
    prefix_tokens=tokenizer.encode(prefix, return_tensors='pt')[0][:-2]
    suffix1=" ".join(spanish_tok.tokenize(suffix)[:3])
    suffix_tokens=tokenizer.encode(suffix1.strip(), return_tensors='pt')[0][:-2].tolist()
    prefix_len=len(prefix_tokens)
    complex_tokens = tokenizer.encode(complex_word.strip())[:-2]
    attn_len = len(prefix_tokens)+len(complex_tokens)
    outputs,scores_with_suffix,scores_with_suffix_masks=model.generate(sentence_tokens.cuda(), 
                            num_beams=beam, 
                            min_length=3,
                            max_length=attn_len+2+20,
                            num_return_sequences=beam,
                            prefix_ids=prefix_tokens,
                            suffix_ids=suffix_tokens,
                            max_aheads=5,
                            tokenizer=tokenizer,
                            complex_len=1,
                            attn_len=None
                        )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        outputs=outputs.cpu()
        scores_with_suffix=scores_with_suffix.cpu()
        scores_with_suffix_masks=scores_with_suffix_masks.cpu()

    outputs,outputs_scores,candis=give_real_scores_ahead(tokenizer,
                                                    outputs,
                                                    scores_with_suffix,
                                                    scores_with_suffix_masks,
                                                    suffix_tokens,
                                                    prefix_len=prefix_len,
                                                    prefix_str=prefix,
                                                    max_ahead=5,
                                                    flag=1)


    if outputs==[]:
        print("find empty substitutes!!!")
        return [],[],[],1
    new_outputs_scores=torch.tensor(outputs_scores)
    outputs_scores=new_outputs_scores
    new_indices=torch.topk(outputs_scores,k=len(outputs_scores),dim=0)[1]
    outputs=[outputs[index1] for index1 in new_indices]
    outputs_scores=[outputs_scores[index1].tolist() for index1 in new_indices]
    candis=[candis[index1] for index1 in new_indices]
    try:
        output_sentences=[" ".join(tokenizer.convert_ids_to_tokens(x)).replace(" ","").replace("▁"," ").strip() for x in outputs]
    except:
        import pdb
        pdb.set_trace()
    for i1 in range(len(output_sentences)):
        output_sentences[i1]=skip_words(output_sentences[i1])

    bertscore_substitutes=[]
    ranking_bertscore_substitutes=[]
    real_prev_scores=[]
    complex_stem = spanish_ps.stem(complex_word)
    not_candi = {"de","que","no","a","la","el","es","en","y","lo","un","por","me","con","tú",\
                    "los","para","como","pero","su","si","al","le","se","del","este","como","esta",".","¡","¿",",","'",'"'}
    not_candi.add(complex_stem)
    not_candi.add(complex_word)

    try:
        not_candi.add(prefix.split()[-1])
    except:
        pass
    try:
        not_candi.add(suffix.split()[0])
    except:
        pass

    all_stored_cand=[]
    for i3 in range(len(candis)):
        candi=candis[i3].lower()
        if "," in candi or "." in candi or "'" in candi or '"' in candi:
            continue
        candi_list=spanish_tok.tokenize(candi)
        max_several=2
        now_several=0
        pointer_several=False
        all_stored_cand.append(candi)


        if len(candi_list)==1:
            candi_stem = spanish_ps.stem(candi)
            not_index_0 = candi.find("-")
            not_index_1 = candi.find(complex_word)
            if candi_stem in not_candi or candi in not_candi or not_index_0 != -1 \
                    or not_index_1 != -1 or candi[0] in punctuation or len(candi)==1 or candi[-1] in punctuation:
                continue

        if candi not in bertscore_substitutes:
            if pointer_several==False:
                bertscore_substitutes.append(candi)
                real_prev_scores.append(outputs_scores[i3])

            elif now_several<max_several:
                bertscore_substitutes.append(candi)
                real_prev_scores.append(outputs_scores[i3])
                now_several+=1
            else:
                pass   
 
    ranking_bertscore_substitutes=bertscore_substitutes 
    
    bertscore_substitutes=bertscore_substitutes[:20]
    ranking_bertscore_substitutes=ranking_bertscore_substitutes[:20]
    real_prev_scores=real_prev_scores[:20]

    freq_scores=cal_freq_scores_spanish(bertscore_substitutes)
    glove_scores_static=cal_static_score(complex_word,bertscore_substitutes)

    real_prev_scores=0.04*torch.tensor(real_prev_scores)+0.02*freq_scores+0.4*glove_scores_static
    real_prev_scores=real_prev_scores.tolist()


    new_real_prev_scores=torch.tensor(real_prev_scores)
    new_indices=torch.topk(new_real_prev_scores,k=len(real_prev_scores),dim=0)[1]    
    bertscore_substitutes=[bertscore_substitutes[index1] for index1 in new_indices]
    ranking_bertscore_substitutes=bertscore_substitutes
    real_prev_scores=[real_prev_scores[index1] for index1 in new_indices]

    bertscore_substitutes=bertscore_substitutes[:10]
    ranking_bertscore_substitutes=ranking_bertscore_substitutes[:10]
    real_prev_scores=real_prev_scores[:10]

    return bertscore_substitutes, ranking_bertscore_substitutes,real_prev_scores,freq_scores


@torch.no_grad()
def evaluate_lexical(model,tokenizer,ori_sentences,completed_steps):
    model.eval()
    potential=0
    f1=open("results/tsar.results.es","w+")
    f2=open("results/tsar.results.es.lookup","w+")
    from tqdm import tqdm
    for i in tqdm(range(len(ori_sentences))):
        line=ori_sentences[i]
        original_text=line.strip().split("\t")[0].strip()

        target_word=line.strip().split("\t")[1].strip()
        label_words=list(set(line.strip().split("\t")[2:]))

        prefix=original_text.split(target_word)[0].strip()
        prefix=prefix.replace("»","").strip()
        prefix=prefix.replace("—","-").strip()

        prefix=" ".join(spanish_tok.tokenize(prefix)).strip()

        again_prefix=" ".join(spanish_tok.tokenize(prefix)).strip()
        if again_prefix!=prefix:
            print("need again prefix")

        prefix=again_prefix
        again_again_prefix=" ".join(spanish_tok.tokenize(prefix)).strip()
        if again_again_prefix!=prefix:
            print("need again and again prefix")
        prefix=again_again_prefix
    
        suffix=original_text.split(target_word)[1].strip()
        suffix=suffix.replace("»","")
        suffix=suffix.replace("—","-")
        if suffix.startswith(")") and prefix.endswith("("):
            suffix=suffix[1:].strip()
            prefix=prefix[:-1].strip()
        else:
            if suffix.startswith("(") or suffix.startswith(")"):
                suffix=suffix[1:].strip()

        if suffix.startswith('"') and prefix.endswith('"'):
            suffix=suffix[1:].strip()
            prefix=prefix[:-1].strip()
        else:
            if suffix.startswith('"'):
                suffix=suffix[1:].strip()
            
        suffix=" ".join(spanish_tok.tokenize(suffix)).strip()
        again_suffix=" ".join(spanish_tok.tokenize(suffix)).strip()
        if again_suffix!=suffix:
            print("need again prefix")

        suffix=again_suffix
        again_again_suffix=" ".join(spanish_tok.tokenize(suffix)).strip()
        if again_again_suffix!=suffix:
            print("need again and again prefix")
        suffix=again_again_suffix


        tmp_original_text=prefix+" "+target_word+" "+suffix
        tmp_original_text=tmp_original_text.strip()

        bert_substitutes, bert_rank_substitutes,real_prev_scores,real_embed_scores=lexicalSubstitute(
            model,
            tokenizer,
            tmp_original_text,
            target_word,
            prefix,
            suffix,
            beam=50
        )

        tp_words=list(set(label_words)&set(bert_substitutes[0:1]))
        if len(tp_words)>0:
            potential+=1

        f1.write(original_text+"\t"+target_word+"\t"+"\t".join(bert_substitutes[:10])+"\n")
        f1.flush()

        tp_words=list(set(label_words)&set(bert_substitutes[:10]))
        f2.write(";".join(label_words)+"|||"+";".join(bert_substitutes)+"|||"+";".join(tp_words)+"\n")
        f2.flush()
        
        
    f1.close()
    f2.close()
    return potential/len(ori_sentences)

evaluate_lexical(model,tokenizer,ori_sentences,1)

