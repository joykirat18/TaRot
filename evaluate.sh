


python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --angle_0 'neg 1/8' --angle_1 '1/8' --dataset operators

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --angle_0 '0' --angle_1 '1/6' --dataset causal_judgment

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --angle_0 '0' --angle_1 '1/6' --dataset elementary_math_qa

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --angle_0 'neg 1/8' --angle_1 '1/8' --dataset operators

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --angle_0 '0' --angle_1 '1/6' --dataset causal_judgment

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --angle_0 '0' --angle_1 '1/6' --dataset elementary_math_qa

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --dataset causal_judgment
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset causal_judgment
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset causal_judgment

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --dataset entailed_polarity
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset entailed_polarity
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset entailed_polarity

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --dataset object_counting
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset object_counting
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset object_counting

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --dataset elementary_math_qa
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset elementary_math_qa
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset elementary_math_qa

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --dataset arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset arithmetic

python3 evaluateRotation.py --model llama-3-8b --layer 0 16 --dataset modified_arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset modified_arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 16 --dataset modified_arithmetic

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset causal_judgment
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset causal_judgment
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset causal_judgment

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset entailed_polarity
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset entailed_polarity
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset entailed_polarity

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset object_counting
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset object_counting
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset object_counting

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset elementary_math_qa
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset elementary_math_qa
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset elementary_math_qa

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset arithmetic

python3 evaluateRotation.py --model llama-3-8b --layer 0 32 --dataset modified_arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset modified_arithmetic
# python3 trainRotation.py --model llama-3-8b --layer 0 32 --dataset modified_arithmetic