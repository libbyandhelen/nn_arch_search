## Neural Network Architecture Search using Gaussian Process and weighted WL kernel

### Main entrance: `main.py`
1. save vectors and labels for stage `i`
2. use soml to optimize weights
3. recalculate dist mat calculated using optimised weights
4. optimize hyper-parameters of gp process
5. run Gaussian Process prdiction
6. sort architectures according to acquisition values
7. eliminate equivalent architectures
8. train candidate models
