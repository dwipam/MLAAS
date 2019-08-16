# MLAAS
Machine Learning As Service with OpenFaas. 

To Start with OpenFaas refer:
```
https://medium.com/@dwipam.katariya/deploying-machine-learning-models-with-openfaas-857f1ba0e9c2?sk=47ba5cfed034b8a034274590833021a2
```

If you already have OpenFaas installed, follow these steps to deploy it to existing stack:

```
faas-cli build -f house-price-model.yml  
faas-cli deploy -f house-price-model.yml
```
