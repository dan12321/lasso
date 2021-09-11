# lasso
Lasso algorithm made in rust. This is a project for my own interest and learning.
​
## Background
Lasso (least absolute shrinkage and selection operator) is a regression method which minimizes the MSE (Mean Squared Error) + the one norm of the vector being approximated. Compared to just minimizing the MSE this encourages values very close to 0 to become 0 which can help produce more interpretable results. 
​
## To Do
- [ ] Add automatically generated test scenarios for large datasets
- [ ] Add cross validation for better testing
- [ ] Profile and improve performance
- [ ] Adaptive step sizes
- [ ] Adapt to Elastic Net