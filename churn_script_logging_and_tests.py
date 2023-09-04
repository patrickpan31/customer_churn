import pytest
import os
import logging
# import churn_library_solution as cls
from churn_library import *

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture
def pth():
	return "./data/bank_data.csv"

def test_import(pth):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data(pth)
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err

@pytest.fixture
def df(pth):
	return import_data(pth)

def test_eda(df):
	'''
	test perform eda function
	'''
	try:
		perform_eda(df)
		img1 = mpimg.imread("./images/eda/churn_plot.png")
		img2 = mpimg.imread("./images/eda/customer_plot.png")
		img3 = mpimg.imread("./images/eda/transaction_plot.png")
		img4 = mpimg.imread("./images/eda/marital_plot.png")
		img5 = mpimg.imread("./images/eda/heatmap_plot.png")
		logging.info('SUCCESS: TESTING EDA, PLOT GENERATED, GOOD!')
	except FileNotFoundError as err:
		logging.error("ERROR: EAD FAILED, NO PLOT GENERATED")
		raise err
    
@pytest.fixture
def cat_lst():
	cat_columns = [
	'Gender',
	'Education_Level',
	'Marital_Status',
	'Income_Category',
	'Card_Category'
]
	return cat_columns

@pytest.fixture
def keep_lst():
	keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
				'Total_Relationship_Count', 'Months_Inactive_12_mon',
				'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
				'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
				'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
				'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
				'Income_Category_Churn', 'Card_Category_Churn']
	return keep_cols

def test_encoder_helper(df, cat_lst, keep_lst):
	'''
	test encoder helper
	'''
	new_df = encoder_helper(df,cat_lst,keep_lst)
	column = new_df.columns.tolist()
	assert all(f'{cat}_Churn' in column for cat in cat_lst)
	logging.info("SUCCESS: ENCODER WORKS")

    
@pytest.fixture
def new_df():
    nedw_data = encoder_helper(df,cat_lst,keep_lst)
    return nedw_data


def test_perform_feature_engineering(new_df):
	'''
	test perform_feature_engineering
	'''
	try:
		X_train, X_test, y_train, y_test = perform_feature_engineering(new_df, None)
		logging.info('SUCCESS: FUNCTION WORKS')
	except:
		logging.error("ERROR: FUNCTION FAILED")
    
	assert X_train.shape[0] == y_train.shape[0]
	assert X_test.shape[0] == y_test.shape[0]

def test_train_models(train_models):
	'''
	test train_models
	'''


# if __name__ == "__main__":
# 	test_import(import_data)
# 	df = import_data("./data/bank_data.csv")
# 	test_eda(perform_eda)








