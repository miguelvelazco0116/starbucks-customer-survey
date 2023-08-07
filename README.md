## Project LoyaltyMax: Enhancing Customer Loyalty with Machine Learning


![loyalty.jpg](loyalty.jpg)


- [Context](#context)
- [Introduction](#introduction)
   + [Data exploring](data_explorer.ipynb)
- [ML models](#ml-models)
   + [building the model](loyalty_drivers.ipynb)


# Context


This dataset comprises survey questions answered by over **100** respondents regarding their buying behavior at **Starbucks**.

The **primary objective** of this project is to **enhance customer loyalty** through the **analysis** of various features such as price, service, product quality, and more. 
By leveraging this data, we aim to provide **insights** into the impact of **investing** in these key areas and how it will influence customer loyalty. Ultimately, the project's output will enable us to make more **accurate pricing decisions** and develop targeted **marketing campaigns** to foster stronger customer loyalty and drive sales growth.


# Introduction


In the upcoming section, it is evident that a significant portion of our client base comprises young females who are either employed or students. Based on this valuable insight, our target audience should be tailored accordingly, and promotional campaigns should align with their preferences. However, several essential questions arise:

  - How should we strategically allocate the budget to achieve maximum impact?
  - Where should our primary focus lie to ensure the effectiveness of our campaigns?
  - If multiple factors come into play, what would be the optimal approach to budget allocation?

By addressing these questions thoughtfully, we can optimize our budget allocation and create impactful campaigns that resonate with our target audience, ultimately driving stronger engagement and loyalty among our customers.


![overview_clients.png](overview_clients.png)


# ML models


Having comprehended the project's context and scope, which revolves around defining the optimal strategy to enhance loyalty, we are now prepared to initiate the modeling process. However, before proceeding further, it is imperative to assess the importance of the features that we intend to evaluate.

Please note that there are additional features to consider, but for the purpose of this example, we will focus on essential aspects such as product quality, price rate, service rate, and other key factors that significantly impact customer loyalty. This preliminary evaluation will provide a solid foundation for our subsequent modeling efforts and enable us to make data-driven decisions to achieve our loyalty enhancement goals.


![radar.png](radar.png)


As observed in the preceding picture, the **price** feature received the lowest rating, while the **promos** and **service** aspects garnered positive feedback. Based on this information, I am inclined to consider a strategy that emphasizes improving the perception of **pricing** while acknowledging the positive response to promotional offers and service.

Additionally, given the (hyphotesis) high **elasticity** in the market, we have an opportunity to leverage **pricing** tactics effectively. By addressing the price perception and offering targeted promotions, we can foster stronger customer engagement and loyalty, positioning our brand more competitively within the market.


### Models


For this example, we will start with four models to predict loyal clients:

- Random Forest
- Logistic Regression
- Support Vector Machine
- Stochastic Gradient Descent

Once we determine the best-performing model, we will proceed to identify the most relevant features that influence the model's predictions. By understanding these key factors, we can develop the most effective strategy to allocate our budget strategically and enhance customer loyalty successfully. This process will enable us to make data-driven decisions and optimize our efforts to achieve the desired loyalty outcomes.


### Evaluation of the models


Upon reviewing the table, we can observe that the models have achieved respectable scores ranging from approximately 0.78 to 0.86. While this metric may suggest that both the logistic regression and random forest models are performing well, it is important to be cautious of potential misleading conclusions.

To ensure an accurate selection, we opt to prioritize the AUC ROC metric for evaluating our model's performance. This involves a meticulous fine-tuning of hyperparameters to make a more informed decision about the optimal model. By employing this rigorous approach, we can confidently choose the best-performing model, enabling us to drive more effective and reliable outcomes in our efforts to enhance customer loyalty.


![models.png](models.png)

```python

```
