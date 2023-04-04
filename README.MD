# MultiLS
Implementation of our paper "Multilingual Lexical Substitution Generation based on
Paraphrase Modeling for Lexical Simplification"
# Dependencies&Installation
This project is mainly buld on [transformers](https://github.com/huggingface/transformers/tree/v4.20-release), with customized modification of scripts.To start, you need to clone this repo and install transformers firstly. Use the following pip command in transformers/:  
pip install -e . 
##other dependencies
[fastText](https://github.com/facebookresearch/fastText)
[wordfreq](https://github.com/rspeer/wordfreq)

# Pretrained model
You need download the following pretrianed models and put them into folder model/
[NLLB](https://huggingface.co/facebook/nllb-200-3.3B)
[fastText embedding](https://fasttext.cc/docs/en/crawl-vectors.html)
# How to run
1)English
sh sh_EN_LS.sh
2)Spanish
sh ES_LS.sh
3)Portuguese
sh_PT_LS.sh

# Citation
<!-- Please cite as: -->
