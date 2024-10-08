# dataset
# modified_arithmetic
# arithmetic
# elementary_math_qa
# operators

# color
# navigate
# entailed_polarity
# ag_news
# winowhy

# model L trainingSize
# llama-3-8b 0-16 10
# phi-3 0-6 20
# qwen_2 0-5 20
# mixtral 0-16 10

python3 train_summarization.py --model phi-3 --layer 0 16 --angle_0 'neg 1/4' --angle_1 '1/4' --dataset 'imdb' --moduleType 'reasoning' --metric 'prob' --fewShotCategory 'mix' --trainingDataSize 12 --rotationMethod 'rotary'
python3 test_summarization.py --model phi-3 --layer 0 16 --angle_0 'neg 1/4' --angle_1 '1/4' --dataset imdb --moduleType 'reasoning' --metric 'prob' --fewShotCategory 'mix' --rotationMethod 'rotary' --testingDataSize 50

python3 train_toxicity.py --model llama-3-8b --layer 0 16 --angle_0 'neg 1/6' --angle_1 '1/6' --dataset 'toxicity' --moduleType 'reasoning' --metric 'prob' --fewShotCategory 'mix' --trainingDataSize 10 --rotationMethod 'rotary'
python3 test_toxicity.py --model llama-3-8b --layer 0 16 --angle_0 'neg 1/6' --angle_1 '1/6' --dataset toxicity --moduleType 'reasoning' --metric 'prob' --fewShotCategory 'mix' --rotationMethod 'rotary' --testingDataSize 50

python3 trainRotation_NN_kernel.py --model llama-3-8b --layer 0 16 --angle_0 'neg 1/6' --angle_1 '1/6' --dataset 'color' --moduleType 'reasoning' --metric 'prob' --fewShotCategory 'mix' --trainingDataSize 10 --rotationMethod 'rotary'
python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --angle_0 'neg 1/6' --angle_1 '1/6' --dataset color --moduleType 'reasoning' --metric 'prob' --fewShotCategory 'mix' --rotationMethod 'rotary'

