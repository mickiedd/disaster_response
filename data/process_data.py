import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - the file path of message dataset (I this you know that by the name)
    categories_filepath - fiel path of categories dataset (see above)
    
    OUTPUT:
    df - dataframe merged from message and categories dataset
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    print(messages.shape)
    print(messages.head())
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    print(categories.shape)
    print(categories.head())
    # merge data
    messages_id = np.array(messages['id'])
    categories_id = np.array(categories['id'])
    len(np.intersect1d(messages_id, categories_id))
    len(np.unique(categories_id))
    # merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')
    print(df.shape)
    df.head()
    # select the first row of the categories dataframe
    row = np.array(categories.loc[0]['categories'].split(";"))
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda x : x.split("-")[0], row))
    print(category_colnames)
    categories = pd.DataFrame(columns=categories.loc[0]['categories'].split(";"))
    categories.head()
    # rename the columns of `categories`
    categories.columns = category_colnames
    categories.head()
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = df['categories'].apply(lambda x : x.split(column + "-")[1].split(";")[0])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    print(categories.shape)
    categories.head()
    # drop the original categories column from `df`
    df = pd.concat([df, categories], axis=1)
    df.head()
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.drop('categories', axis=1)
    df.head()
    return df

def clean_data(df):
    '''
    INPUT:
    df - dataframe that merged from message categories
    
    OUTPUT:
    df - dataframe after data clean
    
    '''
    # check number of duplicates
    unique_indices = np.array(np.nonzero(np.unique(df['id']))).ravel()
    df.shape[0] - len(unique_indices)
    # drop duplicates
    df = df.loc[unique_indices, :]
    # check number of duplicates
    df.shape
    return df

def save_data(df, database_filename):
    '''
    INPUT:
    df - cleaned dataframe that merged from message and categories
    OUTPUT:
    no output
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False)


def main():
    '''
    the main function
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()