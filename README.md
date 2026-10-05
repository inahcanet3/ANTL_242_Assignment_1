# ANTL 242 Assignment 1: Employment Growth by Industry

## Project Overview

This project investigates the employment growth of four industries from 2014-2025:

Technology
Healthcare
Education
Food Services

All of this data was collected from the U.S. Bureau of Labor Statistics.

Our two foundational questions:

1. Which of the four industries (Technology, Healthcare, Education, and Food Services) had the highest employment growth from 2014 to 2025?

2. Are employment levels from the previous year good predictors of overall employment growth in the following year? 

## Synthesis of the Process

To begin, I started with the two foundational questions about employment growth. Since I focused the project on four specific industries, I first had to define the jobs that fall under each industry.

Once each industry was defined, I used the employment data from 2014-2025 and processed that data into a SQLite database named employment.db.

On main.py, I pulled the data from the database. This is where I calculated the year-to-year employment growth, as well as compared the growth from year to year.

From there, I wanted to see if the previous year's employment levels could be a predictor for future employment growth. So I split the data for Question 1 and used the data from 2015-2020 as training data, and then used 2021-2025 as my testing data.

For Question 2, I used 2015-2022 as training data and 2023-2025 as testing data.

I used R², MAE, and RMSE to analyze the models.

I found that Healthcare had the highest overall growth in employment and that, in this case with this data, the previous-year employment was a poor predictor of future employment growth.

## Synthesis of Data Gathering

I used the Bureau of Labor Statistics (BLS) Quarterly Census of Employment and Wages (QCEW). From there, I gathered all the annual industry files from 2014-2025. I made sure to gather the necessary NAICS industry codes for the four industries, which was important because I had to make sure that each industry was clearly defined.

Classifications were organized.

Healthcare - NAICS 62

Education - NAICS 61

Food Services - NAICS 722

I had to combine a few employment values for Technology.

Once that was done, I organized the information annually and by each of the four industries.

## AI Use/Difficulties

I used AI to help troubleshoot Python errors and explain the error messages. I also used AI to help me understand the process of how to pull data from databases and which codes I needed to help with the modeling.

I had difficulties with knowing which codes to use and sometimes what to do next. So AI really helped me with just sometimes explaining the process or what the next approaches might be, which helped me understand the process more as well as be more intentional with the answers that I was looking for.

I did have difficulties with just trying to find the correct data and then filtering out all the other files that were part of the database. So it was really interesting because there is so much information on BLS, but a lot of the data I didn't need.

From there, I just cleaned the data, organized the data, and ran tests on the data with the information I extracted in order to find an answer to my questions.
