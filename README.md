# Prone-VGG19
To prone VGG19
## Baseline
```python
python main.py --dataset cifar10 --arch vgg --depth 19
```

## Prune

```python
python vggprone.py --dataset cifar10 --depth 19 --model ./logs/model_best.pth.tar --save ./proned
```

