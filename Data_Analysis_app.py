import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import time

st.header("Do Data Analysis WithOut Libararies")

st.image("data_ana.jpg",use_column_width=True)

import base64

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to your image
img_path = "grey_color.jpg"
img_base64 = get_base64_of_bin_file(img_path)

# Apply CSS to Streamlit's main container to display the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# important variables
categorical_columns = []  # Store the Categorical Column Name
numerical_columns = []    # Store the Numerical Column Name
missing_null_values = []  # storing the missing values


# Upload a File
uploaded_file = st.file_uploader("Upload a Csv File",type='CSV')
if uploaded_file:
    with st.spinner('Loading File'):
        time.sleep(10)
    st.success(f'Successfully Upload a File : {uploaded_file.name}')

if uploaded_file is not None:
    data_frame = pd.read_csv(uploaded_file)
    user_data_frame = data_frame.copy()
    for i in data_frame.columns:
        if data_frame[i].dtype == "object":
            categorical_columns.append(i)
        else:
            numerical_columns.append(i)

    st.subheader(f'{uploaded_file.name} : DataFrame')
    st.dataframe(data_frame)

    # Statistcial Data
    statistical_options = ['Column Names','Statistics Data','Data Types','Null Values']
    statistical_user_data = st.multiselect('Basic Information About Data Frame',options=statistical_options)
    if 'Column Names' in statistical_user_data:
        columns = data_frame.columns
        st.write(f'Columns in  a {uploaded_file.name} File :')
        st.write(columns)
    if 'Statistics Data' in statistical_user_data:
        description = data_frame.describe()
        st.write(f'Statistics in a {uploaded_file.name} File')
        st.write(description)
    if 'Data Types' in statistical_user_data:
        data_types = data_frame.dtypes
        st.write(f'Data Types in a {uploaded_file.name} File')
        st.write(data_types)
    if 'Null Values' in statistical_user_data:
        missing_null_values = data_frame.isnull().sum()
        st.write(f'Null Values in Each Column of Data Frame : {uploaded_file.name}')
        st.write(data_frame.isnull().sum())


    # Finding the Unique Values in the Entire DataFrame
    st.subheader('Identifing Unique Names in Each Categorical Column')
    unique_column_options = []
    entire_data_frame = []
    for i in data_frame.columns:
        if data_frame[i].dtype == "object":
            unique_column_options.append(i)
            entire_data_frame.append({i:data_frame[i].unique()})
    unique_column_options.append('Entire DataFrame')
    unique_column_user = st.multiselect('Unique Names in Columns :',options=unique_column_options)
    if 'Entire DataFrame' in unique_column_user:
        st.write(f'Unique Names for Each Column in the DataFrame : {uploaded_file.name}')
        st.dataframe(entire_data_frame)
        unique_column_user.remove('Entire DataFrame')
    if unique_column_user is not None:
        for i in unique_column_user:
            st.write(f'Unique Names in Column : {i}')
            st.dataframe(data_frame[i].unique())
            st.write(f'Sum of Unique Names in Column : {i} is {data_frame[i].nunique()}')
    
    ## Handling Missing Values
    st.subheader('Handling Missing Null Values')
    if missing_null_values is not None:
        missing_user_data = st.selectbox('Handling Missing Values', options=['Categorical Data', 'Numerical Data'])
        missing_values_info = []

        if missing_user_data == 'Categorical Data':
            st.write('Missing Value Columns in Categorical Data')
            for col in categorical_columns:
                null_count = data_frame[col].isnull().sum()
                if null_count > 0:
                    percentage = (null_count / data_frame.shape[0]) * 100
                    missing_values_info.append({'Column': col, 'Count': null_count, 'Percentage': percentage})

            if missing_values_info:
                st.table(pd.DataFrame(missing_values_info))
                choose_col = st.selectbox('Choose a Column:', options=[info['Column'] for info in missing_values_info])
                if choose_col:
                    method = st.selectbox("Choose a Method:", options=['Most Frequent', 'Constant', 'Drop the Values'])
                    if method == 'Most Frequent':
                        data =  data_frame[choose_col].fillna(data_frame[choose_col].mode()[0])
                        with st.expander('Seeing the DataFrame After Applying Operation Method : Fill Mode Frequency'):
                            st.dataframe(data)
                        # user_defined data_frame modifications
                        checkbox = st.checkbox('Upadate the Values in user_defined Data Frame')
                        if checkbox:
                            user_data_frame[choose_col].fillna(data_frame[choose_col].mode()[0],inplace=True)
                    elif method == 'Constant':
                        const_value = st.text_input('Enter the Constant Value:')
                        data = data_frame[choose_col].fillna(const_value)
                        with st.expander('Seeing the DataFrame After Applying Operation Method : Fill Constant'):
                            st.dataframe(data)
                        checkbox = st.checkbox('Upadate the Values in User_defined DataFrame')
                        if checkbox:
                            user_data_frame[choose_col].fillna(const_value,inplace=True)
                    elif method == 'Drop the Values':
                        data = data_frame.drop(columns=choose_col)
                        with st.expander('Seeing the DataFrame After Applying Operation Method : Drop Values'):
                            st.dataframe(data)
                        checkbox = st.checkbox('Updatae the Values in User Defined DataFrame')
                        if checkbox:
                            user_data_frame[choose_col].dropna(inplace=True)
                    with st.expander('Display the User Defined DataFrame After Handling Misssing Values'):
                        st.dataframe(user_data_frame)
            else:
                st.success('No Missing Values are There in Categorical Data')
        if missing_user_data == 'Numerical Data':
            st.write('Missing Value Columns in Numerical Data')
            for col in numerical_columns:
                null_count = data_frame[col].isnull().sum()
                if null_count > 0:
                    percentage = (null_count / data_frame.shape[0]) * 100
                    missing_values_info.append({'Column': col, 'Count': null_count, 'Percentage': percentage})

            if missing_values_info:
                st.table(pd.DataFrame(missing_values_info))
                choose_col = st.selectbox('Choose a Column:', options=[info['Column'] for info in missing_values_info])
                if choose_col:
                    method = st.selectbox("Choose a Method:", options=['Mean', 'Median', 'Mode', 'Constant', 'Drop the Values(Rows)','Drop Entire Column'])
                    if method == 'Mean':
                        data = data_frame[choose_col].fillna(data_frame[choose_col].mean())
                        with st.expander('Show the DataFrame After Applying Operating Fill - Mean'):
                            st.dataframe(data)
                        checkbox = st.checkbox('Upadata the User Defined DataFrame Numerical Missing Values with Mean')
                        if checkbox:
                            user_data_frame[choose_col].fillna(user_data_frame[choose_col].mean(),inplace=True)
                    elif method == 'Median':
                        data = data_frame[choose_col].fillna(data_frame[choose_col].median())
                        with st.expander('Show the DataFrame After Applying Operating Fill - Median'):
                            st.dataframe(data)
                        checkbox = st.checkbox('Upadata the User Defined DataFrame Numerical Missing Values with Median')
                        if checkbox:
                            user_data_frame[choose_col].fillna(user_data_frame[choose_col].median(),inplace=True)
                    elif method == 'Mode':
                        data = data_frame[choose_col].fillna(data_frame[choose_col].mode()[0])
                        with st.expander('Show the DataFrame After Applying Operating Fill - Mode'):
                            st.dataframe(data)
                        checkbox = st.checkbox('Upadata the User Defined DataFrame Numerical Missing Values with Most Frequenct values')
                        if checkbox:
                            user_data_frame[choose_col].fillna(user_data_frame[choose_col].mode()[0],inplace=True)
                    elif method == 'Constant':
                        const_value = st.number_input('Enter a Constant Value:')
                        data = data_frame[choose_col].fillna(const_value)
                        with st.expander('Show the DataFrame After Applying Operating Fill - Constant Value'):
                            st.dataframe(data)
                        checkbox = st.checkbox('Upadata the User Defined DataFrame Numerical Missing Values with Constant')
                        if checkbox:
                            user_data_frame[choose_col].fillna(const_value,inplace=True)
                    elif method == 'Drop the Values(Rows)':
                        # Drop rows with NaN values
                        data = data_frame.dropna()
                        with st.expander('Show the DataFrame After Applying Operation - Drop the Rows'):
                            st.dataframe(data)
                        
                        # Checkbox to update the user-defined DataFrame
                        checkbox = st.checkbox('Update the User-Defined DataFrame: Rows with Missing Values Dropped')
                        if checkbox:
                            user_data_frame.dropna(inplace=True)

                    elif method == 'Drop Entire Column':
                        # Drop selected column
                        data = data_frame.drop(columns=choose_col)
                        with st.expander('Show the DataFrame After Applying Operation - Drop the Column'):
                            st.dataframe(data)
                        
                        # Checkbox to update the user-defined DataFrame
                        checkbox = st.checkbox('Update the User-Defined DataFrame: Selected Columns with Missing Values Dropped')
                        if checkbox:
                            user_data_frame.drop(columns=choose_col, inplace=True)

                    # Show the updated user-defined DataFrame
                    with st.expander('Show the Updated User-Defined DataFrame'):
                        st.dataframe(user_data_frame)

            else:
                st.success('No Missing Values are there in Numerical Data')
        
    
    # Selecting and Indexing of the Columns
    st.subheader('Making DataFrame By Selecting Columns')
    all_column_names = user_data_frame.columns
    user_select_columns = st.multiselect('Select Columns to to Make Data Frame :',options = all_column_names)
    if user_select_columns:
        st.dataframe(user_data_frame[user_select_columns])
                   

    ## Creating User_categorical Column and user_Numerical Column
    user_categorical_columns = []
    user_numerical_columns = []
    for i in user_data_frame.columns:
        if data_frame[i].dtype == "object":
            user_categorical_columns.append(i)
        else:
            user_numerical_columns.append(i)
    ## Filtering the DataFrame using Conditions
    st.header('Filtering the Data') 
    st.subheader('Select Categorical Column to Filter Data')
    if categorical_columns is not None:  # Categorical Columns
        user_select_categorical_columns = st.multiselect('Select Columns :',options=user_categorical_columns)
        data_columns = user_select_categorical_columns
        data = user_data_frame.copy()
        if user_select_categorical_columns:
            select_inneer_column = False
            for i in user_select_categorical_columns:
                option_values = data[i].unique()
                user_select_data = st.multiselect(f'Select You Want in columns {i}',options=option_values)
                if user_select_data is not None:
                    select_inneer_column = True
                data = data[data[i].isin(user_select_data)]
            if select_inneer_column == True:
                with st.expander('DataFrame After Filtering Categorical Data With Conditios'):
                    st.dataframe(data)
        else:
            st.write('Please Select Columns to Filter Data')
    if numerical_columns is not None:
        st.subheader('Select Numerical Columns to filter Data')
        user_select_numerical_columns = st.multiselect('Select Numerical Columns to Filter the Data',options=numerical_columns)
        if user_select_numerical_columns:
            for i in user_select_numerical_columns :
                st.write(f'Selct a Range for Column_name :  {i}')
                min_val,max_value = st.slider('Select range ',min_value=int(data[i].min()),max_value = int(data[i].max()),value=(int(data[i].min()), int(data[i].max())))
                data = data[(data[i]>=min_val) & (data[i]<=max_value)]
                if min_val == 0 and max_value == 0:continue
            with st.expander('DataFrame After Filtering Numerical Data with Conditions'):
                st.dataframe(data)
        else:
            st.write('Please Select Columns Filter Data ')


    ## Filtering the Duplicate Values
    st.subheader("Filtering the Duplicate Values")
    user_select_duplicate_columns = st.multiselect('Choose Column for Duplicates :',options=user_data_frame.columns)
    if user_select_duplicate_columns:
        keep_duplicates = st.selectbox('Duplicates Operations :',options=['Keep First Values','Keep Last Values','Drop all the Value'])
        if keep_duplicates == "Keep First Values":
            st.write('Drop the Duplicate value in last Occurences')
            duplicate_rows = user_data_frame.drop_duplicates(subset=user_select_duplicate_columns)
            with st.expander('Duplicate Value for Give Columns Method : Last Occurences'):
                st.dataframe(duplicate_rows)
            checkbox = st.checkbox('Update user DataFrame by Deleting Duplicate Rows')
            if checkbox:
                user_data_frame.drop_duplicates(subset=user_select_duplicate_columns,keep='first')
        elif keep_duplicates == 'Keep Last Values':
            st.write('Drop Duplicate Value in First Occurences')
            duplicate_rows = user_data_frame.drop_duplicates(subset=user_select_duplicate_columns)
            with st.expander('Duplicate Value for Give Columns Method : First Occurences'):
                st.dataframe(duplicate_rows)
            checkbox = st.checkbox('Update User Defined DataFrame By Deleting Duplicate Rows')
            if checkbox:
                user_data_frame.drop_duplicates(subset=user_select_duplicate_columns,keep='last')
        elif keep_duplicates == 'Drop all the Value':
            st.write('Drop all the Duplicates rows in the Data Frame')
            duplicate_rows = user_data_frame.drop_duplicates(subset=user_select_duplicate_columns)
            with st.expander('Duplicate Value for Give Columns Method : All Occurences'):
                st.dataframe(duplicate_rows)
            checkbox = st.checkbox('Updating User Defined DataFrame By Deleting Duplicated Rows')
            if checkbox:
                user_data_frame.drop_duplicates(subset=user_select_duplicate_columns)
        with st.expander('Show The Data Frame After Deleting Duplicate Rows'):
            st.dataframe(user_data_frame)

    
    # Sorting the Columns
    st.header('Sorting Data Frame')
    sort_dataframe = st.multiselect('Select Columnt to Sort the Data',options=user_data_frame.columns)
    if sort_dataframe:
        sorting_optins = st.selectbox('choose sorting Methods :',options=['Increase','Decrease'])
        if sorting_optins is not None:
            if sorting_optins == 'Increase':
                user_data_frame.sort_values(by=sort_dataframe,ascending = True,inplace=True)
            elif sorting_optins == 'Decrease':
                user_data_frame.sort_values(by=sort_dataframe,ascending=False,inplace=True)
            with st.expander('Sorting DataFrame'):
                st.dataframe(user_data_frame)


    ## Data Grouping and Aggregations
    st.header('Data Grouping and Aggregation')
    st.subheader('Data Grouping and Aggregation in Categorical Column')

    # User selects columns for grouping
    user_select_grouping = st.multiselect('Select grouping Columns:', options=user_categorical_columns)

    if user_select_grouping:
        # Perform grouping based on selected columns
        data_group = user_data_frame.groupby(user_select_grouping)
        
        # User selects columns for aggregation
        user_select_after_grouping_columns = st.multiselect(' Select Extracting Columns', options=user_data_frame.columns)
        
        if user_select_after_grouping_columns:
            # User selects aggregation method
            user_select_methods = st.selectbox(' Select Method:', options=['Count', 'nunique', 'size', 'count Values'])
            
            # Perform the selected aggregation method
            if user_select_methods:
                if user_select_methods == 'Count':
                    count = data_group[user_select_after_grouping_columns].count()
                    with st.expander('Data Frame After Applying Count Method'):
                        st.dataframe(count)
                
                elif user_select_methods == 'nunique':
                    Nunique = data_group[user_select_after_grouping_columns].nunique()
                    with st.expander('DataFrame After Applying NUnique Method'):
                        st.dataframe(Nunique)
                
                elif user_select_methods == 'size':
                    size = data_group.size()
                    with st.expander('DataFrame After Applying Size Method'):
                        st.dataframe(size)
                
                elif user_select_methods == 'count Values':
                    # value_counts() needs to be called on individual columns, not on grouped DataFrame
                    count_values = user_data_frame[user_select_after_grouping_columns].apply(lambda x: x.value_counts())
                    with st.expander('DataFrame After Applying Count Values Method :'):
                        st.dataframe(count_values)
    else:
        st.write('Please select columns to group the data.')

    
    # Data Grouping and Aggregation in Numerical Columns
    st.subheader('Data Grouping and Aggregation in Numerical Columns')
    user_select_grouping_numerical = st.multiselect('select Numerical Columns for Aggregation :',options=user_data_frame.columns)
    if user_select_grouping_numerical:
        user_select_column_after_grouping = st.multiselect('Please select Columns :',options=user_numerical_columns)
        numerical_data = user_data_frame.groupby(user_select_grouping_numerical)[user_select_column_after_grouping]
        if user_select_column_after_grouping:
            methods = ['Sum','Mean','Max','Standard Deviation','Variance','CumSum','CumProduct']
            user_methods_select = st.selectbox('Please select Methods :',options=methods)
            if user_methods_select:
                if user_methods_select == 'Sum':
                    data = numerical_data.sum()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Sum'):
                        st.dataframe(data)
                elif user_methods_select == 'Mean':
                    data = numerical_data.mean()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Mean'):
                        st.dataframe(data)
                elif user_methods_select == 'Max':
                    data = numerical_data.max()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Max'):
                        st.dataframe(data)
                elif user_methods_select == 'Standard Deviation':
                    data = numerical_data.std()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Standard Deviation'):
                        st.dataframe(data)
                elif user_methods_select == 'Variance':
                    data = numerical_data.var()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Variance'):
                        st.dataframe(data)
                elif user_methods_select == 'CumSum':
                    data = numerical_data.cumsum()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Cummulative Sum'):
                        st.dataframe(data)
                elif user_methods_select == 'CumProduct':
                    data = numerical_data.cumprod()
                    with st.expander('DataFrame After Applying Aggreagation with Method : Cummulative Product'):
                        st.dataframe(data)
    else:
        st.write('Please Select Columns')

    ### Creating a new Columns
    st.subheader('Creating New Columns')
    creating_column = True
    if creating_column:
        column_name = st.text_input('Enter Column Name :')
        if column_name:
            column_methods = ['Constant','Arithmetic Operations','Arithmetic Functions','Conditions','String Functions']
            creating_methods = st.selectbox('Please Select Methods',options=column_methods)
            if creating_methods:
                if creating_methods == "Constant":
                    constant_method = ['Constant','Multiplication','Division','Addition','Subtraction','Power Transformation']
                    column_selection = st.selectbox('Please select Column :',options=numerical_columns)
                    if column_selection:
                        user_methods = st.selectbox('Please Select constant Method :',options=constant_method)
                        if user_methods:
                            number = st.number_input('Enter a Constant Value')
                            if number:
                                if user_methods == 'Constant':
                                    user_data_frame[column_name] = number
                                    with st.expander('DataFrame After Adding a Column with Constant Value'):
                                        st.dataframe(user_data_frame)
                                elif user_methods == 'Multiplication':
                                    user_data_frame[column_name] = user_data_frame[column_selection] * number
                                    with st.expander('DataFrame Adding a Column with Method : Column * Constant'):
                                        st.dataframe(user_data_frame)
                                elif user_methods == 'Division':
                                    user_data_frame[column_name] = user_data_frame[column_selection]/number
                                    with st.expander('DataFrame Adding a Column with Method : Column / Constant'):
                                        st.dataframe(user_data_frame)
                                elif user_methods == 'Addition':
                                    user_data_frame[column_name] = user_data_frame[column_selection] + number
                                    with st.expander('DataFrame Adding a Column with Method : Column + Constant'):
                                        st.dataframe(user_data_frame)
                                elif user_methods == 'Subtraction':
                                    user_data_frame[column_name] = user_data_frame[column_selection] - number
                                    with st.expander('DataFrame Adding a Column with Method : Column - Constant'):
                                        st.dataframe(user_data_frame)
                                elif user_methods == 'Power Transformation':
                                        user_data_frame[column_name] = user_data_frame[column_selection]**number
                                        with st.expander('DataFrame Adding a Column with Method : Column ** Constant'):
                                            st.dataframe(user_data_frame)
                elif creating_methods == 'Arithmetic Operations':
                    column_select = st.multiselect('Please Select Columns',options=user_numerical_columns)
                    if len(column_select) >=2:
                        artithmethic_methods = ['Additions','Multiplication','Division','Divider','Subtraction']
                        arthimetic_select = st.selectbox('Select Arithmetic Operations :',options=artithmethic_methods)
                        if arthimetic_select == 'Additions':
                            user_data_frame[column_name] = 0
                            for i in column_select:
                                user_data_frame[column_name] += user_data_frame[i]
                            with st.expander('DataFrame After Adding Column  Method : Adding Columns'):
                                st.dataframe(user_data_frame)
                        elif arthimetic_select == 'Multiplication':
                            user_data_frame[column_name] = 1
                            for i in column_select:
                                user_data_frame[column_name] *= user_data_frame[i]
                            with st.expander('DataFrame After Adding Column  Method : Multiplying Columns'):
                                st.dataframe(user_data_frame)
                        elif arthimetic_select == 'Division':
                            user_data_frame[column_name] = user_data_frame[column_select[0]]
                            for i in range(1,len(column_select)):
                                user_data_frame[column_name] /= user_data_frame[column_select[i]]
                            with st.expander('DataFrame After Adding Column  Method : Division Columns'):
                                st.dataframe(user_data_frame)
                        elif arthimetic_select == 'Divider':
                            user_data_frame[column_name] = user_data_frame[column_select[0]]
                            for i in range(1,len(column_select)):
                                user_data_frame[column_name] %= user_data_frame[column_select[i]]
                            with st.expander('DataFrame After Adding Column  Method : Divider Columns'):
                                st.dataframe(user_data_frame)
                        elif arthimetic_select == 'Subtraction':
                            user_data_frame[column_name] = user_data_frame[column_select[0]]
                            for i in range(1,len(column_select)):
                                user_data_frame[column_name] -= user_data_frame[column_select[i]]
                            with st.expander('DataFrame After Adding Column  Method : Subtracting Columns'):
                                st.dataframe(user_data_frame)
                    else:
                        st.write('Select atleast Two Columns')
                elif creating_methods == 'Arithmetic Functions':
                    select_column = st.selectbox('Please Select Column',options=user_numerical_columns)
                    if select_column:
                        methods = ['Logarithm','Exponential','Square-Root']
                        select_method = st.selectbox('Select Method :',options=methods)
                        if select_method:
                            if select_method == 'Logarithm':
                                user_data_frame[column_name] = np.log(user_data_frame[select_column])
                                with st.expander('DataFrame After Adding Column Method : Logarithm'):
                                    st.dataframe(user_data_frame)
                            if select_method == 'Exponential':
                                user_data_frame[column_name] = np.exp(user_data_frame[select_column])
                                with st.expander('DataFrame After Adding Column Method : Exponential'):
                                    st.dataframe(user_data_frame)
                            if select_method == 'Square-Root':
                                user_data_frame[column_name] = np.sqrt(user_data_frame[select_column])
                                with st.expander('DataFrame After Adding Column Method : Square-Root'):
                                    st.dataframe(user_data_frame)
                elif creating_methods == 'Conditions':
                    select_columnss = st.selectbox('Select Column :',options=numerical_columns)
                    if select_columnss:
                        conditon_value = st.number_input('Enter Condition Value')
                        condition_text1 = st.text_input('Enter a Text. If Value is Less')
                        condition_text2 = st.text_input('Enter a Text. If Value is Greater')
                        if conditon_value and condition_text1 and condition_text2 :
                            user_data_frame[column_name] = np.where(user_data_frame[select_columnss]>conditon_value,condition_text1,condition_text2)
                            with st.expander('DataFrame After Adding Column Method : Condition'):
                                    st.dataframe(user_data_frame)
                    
                elif creating_methods == 'String Functions':
                    string_columns = ['Extracting','UpperCase','LowerCase','Length']
                    select_columns = st.selectbox('Please Select Column',options=user_categorical_columns)
                    if select_columns:
                        string_methods = st.selectbox('Please Select String Methods',options=string_columns)
                        if string_methods:
                            if string_methods == 'Extracting':
                                start_value = st.number_input('Enter Starting Index', min_value=0, value=0)
                                ending_value = st.number_input('Enter Ending Index', min_value=0, value=len(data_frame[select_columns][0]))

                                # Check if ending_value is greater than the maximum length of strings in the selected column
                                max_length = data_frame[select_columns].str.len().max()
                                if ending_value > max_length:
                                    ending_value = max_length

                                # Ensure ending_value is greater than start_value
                                if start_value >= ending_value:
                                    st.error("Ending Index must be greater than Starting Index.")
                                else:
                                    # Slicing the string column based on user inputs
                                    user_data_frame[column_name] = user_data_frame[select_columns].str[start_value:ending_value]
                                    
                                    # Checkbox to show the modified DataFrame
                                    with st.expander('DataFrame After Adding a Column Method : string Extraction'):
                                        st.dataframe(user_data_frame)
                            elif string_methods == 'UpperCase':
                                user_data_frame[column_name] = user_data_frame[select_columns].str.upper()
                                with st.expander('DataFrame After Adding Column Method : String UpperCase'):
                                    st.dataframe(user_data_frame)
                            elif string_methods == 'LowerCase':
                                user_data_frame[column_name] = user_data_frame[select_columns].str.lower()
                                with st.expander('DataFrame After Adding Column Method : String LowerCase'):
                                    st.dataframe(user_data_frame)
                            elif string_methods == 'Length':
                                user_data_frame[column_name] = user_data_frame[select_columns].str.len()
                                with st.expander('DataFrame After Adding Column Method : String Length'):
                                    st.dataframe(user_data_frame)
                        
            else:
                st.write('Please Select required Methods')             
        
    
    ## Important Information About Charts
    st.subheader('Important Information About Charts')
    with st.expander('Important Information About Charts'):
        
        st.write("Bar-Chart Description : Compares quantities across categories using bars; useful for comparing discrete data points.")
        
        st.write("Line-Chart Description : Shows trends over time by connecting data points with a continuous line, great for time-series data.")
        
        st.write("Histogram-Chart Description : Displays the distribution of a dataset by grouping values into ranges, often used for frequency analysis.")
        
        st.write("Pie-Chart Description : Represents parts of a whole as segments of a circle, effective for showing proportions.")
        
        st.write("Scatter-Plot Description : Visualizes the relationship between two continuous variables, revealing correlations and data clustering.")
        
        st.write("Box-Plot Description : Summarizes the distribution of a dataset through quartiles, highlighting median and potential outliers.")
        
        st.write("Heat Map Description: Uses color to represent values in a matrix, ideal for visualizing patterns or correlations in complex datasets.")
        
        st.write("Area-Chart Descripton : Similar to a line chart but with filled areas under the line, useful for showing cumulative data over time.")
        
        st.write("Bubble-Chart Descripton : Extends scatter plots with bubble size to show an additional dimension, useful for multi-variable comparison")
        
        st.write("Funnel-Chart Description : Shows the progressive reduction of data as it passes through stages, commonly used in sales and marketing to illustrate conversion rates.")
        
        st.write("Violin-Plot Description : Combines a box plot and density plot to show the distribution of data, often used to compare multiple groups")
       
        st.write("Density-Plot Description : Shows the distribution of a continuous variable, similar to a histogram but with a smoother curve.")

    # Charts in Data Analysis
    st.subheader('Data-Analysis Charts')

    # Multiselect for columns
    
    charts_columns_categorical = st.multiselect('Please Select Categorical Columns:', options=user_categorical_columns)
    charts_columns_numerical = st.multiselect('Please Select Numerical Columns:', options=user_numerical_columns)

    # Initialize the chart options list
    chart_columns_select = []

    # Set the chart selection conditions
    categorical_columns_size = len(charts_columns_categorical)
    numerical_columns_size = len(charts_columns_numerical)

    # Add chart types based on column selections
    if categorical_columns_size >= 0 and numerical_columns_size >= 1:
        chart_columns_select.extend(['Bar Chart', 'Line Chart', 'Box Plot', 'Area Chart', 
                                    'Violin Plot', 'Boxen Plot', 
                                    'Density Plot', 'Funnel Chart', 'WaterFall Chart','Scatter Plot'])
    if categorical_columns_size >= 1 and numerical_columns_size >= 1:
        chart_columns_select.extend(['Bubble Chart'])
    if categorical_columns_size >= 1 and numerical_columns_size >= 1:
        chart_columns_select.extend(['Heat Map'])
    if categorical_columns_size == 0 and numerical_columns_size >= 2:
        chart_columns_select.extend(['Co-relaton Chart(Heat Map)'])
    if categorical_columns_size == 1 and numerical_columns_size == 1:
        chart_columns_select.append('Histogram')
    if categorical_columns_size == 1 and numerical_columns_size == 0:
        chart_columns_select.append('Count Plot')
    if categorical_columns_size ==1 and numerical_columns_size == 1:
        chart_columns_select.append('Pie Chart')

    # User selects charts to display
    user_select_charts = st.multiselect('Chart are Available for the Combination of Numerical and Categorical Column:', options=chart_columns_select)

    # Check if the user selected any chart type
    if user_select_charts:
        # If Bar Chart is selected
        if 'Bar Chart' in user_select_charts:
            
            melted_df = user_data_frame.melt(id_vars=charts_columns_categorical, value_vars=charts_columns_numerical,var_name='Numerical Variable', value_name='Value')
            # Create a combined categorical variable for better grouping
            melted_category = melted_df[charts_columns_categorical[0]]
            for i in range(1,len(charts_columns_categorical)):
                melted_category = melted_category + '-' + melted_df[charts_columns_categorical[i]]
            melted_df['Combined Category'] = melted_category

            # Create a grouped bar chart
            plt.figure(figsize=(12, 6))
            sns.barplot(data=melted_df, x='Combined Category', y='Value', hue='Numerical Variable')

            # Customize the plot
            plt.title('Grouped Bar Chart for Multiple Numerical and Categorical Columns')
            plt.xlabel('Combined Categories')
            plt.ylabel('Values')
            plt.legend(title='Numerical Variables')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
            with st.expander('Bar Chart'):
                st.pyplot(plt.gcf())


        if 'Pie Chart' in user_select_charts:
            # Ensure only one categorical and one numerical column are selected for simplicity
            if len(charts_columns_categorical) == 1 and len(charts_columns_numerical) == 1:
                categorical_col = charts_columns_categorical[0]
                numerical_col = charts_columns_numerical[0]

                # Aggregate the numerical data by the categorical column
                agg_data = user_data_frame.groupby(categorical_col)[numerical_col].sum().reset_index()

                # Extract the values and labels for the pie chart
                values = agg_data[numerical_col]
                labels = agg_data[categorical_col]

                # Plot the pie chart
                plt.figure(figsize=(12, 6))
                plt.pie(values, labels=labels, autopct="%1.1f%%")
                plt.title(f'{categorical_col} vs {numerical_col} : Pie Chart')

                # Display the pie chart in Streamlit
                with st.expander('Pie Chart'):
                    st.pyplot(plt.gcf())
            else:
                st.warning("Please select one categorical column and one numerical column for the pie chart.")
        
        # Line Chart
        if 'Line Chart' in user_select_charts:
            # Melt Data
            melt_data = pd.melt(user_data_frame,id_vars=charts_columns_categorical,value_vars=charts_columns_numerical,var_name='Categorical Data',value_name='Numerical Values')
            plt.figure(figsize=(12,6))
            sns.lineplot(data=melt_data,x='Categorical Data',y='Numerical Values',palette='viridis')
            plt.legend()
            with st.expander('Line Chart'):
                st.pyplot(plt.gcf())

        ## Box plot using seaborn
        if 'Box Plot' in user_select_charts:
            ## Melt data
            melt_data = pd.melt(user_data_frame,id_vars=charts_columns_categorical,value_vars=charts_columns_numerical,var_name='Combined Category',value_name='Numerical Values')
            melted_category = user_data_frame[charts_columns_categorical[0]]
            for i in range(1,len(charts_columns_categorical)):
                melted_category = melted_category + '-' + melt_data[charts_columns_categorical[i]]
            melt_data['Combined Category'] = melted_category

            plt.figure(figsize=(12,6))
            sns.boxplot(melt_data,x='Combined Category',y='Numerical Values',palette='viridis')
            plt.xticks(rotation = 70)
            with st.expander('Box Plot'):
                st.pyplot(plt.gcf())

    ## Area Chart using seaborn
    
    ## Violin Plot
    if 'Violin Plot' in user_select_charts:
        ## Melting the Data for multiple categorical columns and multiple numerical columns
        melted_data = pd.melt(user_data_frame,id_vars=charts_columns_categorical,value_vars=charts_columns_numerical,var_name='Categorical Data',value_name='Numerical Data')

        melted_category = melted_data[charts_columns_categorical[0]]
        for i in range(1,len(charts_columns_categorical)):
            melted_category = melted_category + '_' + melted_data[charts_columns_categorical[i]]
        melted_data['Categorical Data'] = melted_category
        plt.figure(figsize=(12,6))
        sns.violinplot(data=melted_data,x='Categorical Data',y='Numerical Data',palette='viridis')
        plt.xticks(rotation=70)
        with st.expander('Violin Plot'):
            st.pyplot(plt.gcf())

    ## Boxen- Plot
    if 'Boxen Plot' in user_select_charts:
        ## Melting Data for multiple categorical columns and multiple numerical columns
        melted_data = pd.melt(user_data_frame,id_vars=charts_columns_categorical,value_vars=charts_columns_numerical,var_name='Categorical Data',value_name='Numerical Data')
        melted_category = melted_data[charts_columns_categorical[0]]
        for i in range(1,len(charts_columns_categorical)):
            melted_category = melted_category + '_' + melted_data[charts_columns_categorical[i]]
        melted_data['Categorical Data'] = melted_category
        plt.figure(figsize=(12,6))
        sns.boxenplot(melted_data,x='Categorical Data',y='Numerical Data',palette='coolwarm')
        plt.xticks(rotation=70)
        with st.expander('Boxen Plot'):
            st.pyplot(plt.gcf())
    
    ## Density Plot
    if 'Density Plot' in user_select_charts:
        ## Melting Data for multiple categorical columns and multiple numerical columns
        melted_data = pd.melt(user_data_frame,id_vars=charts_columns_categorical,value_vars=charts_columns_numerical,var_name='Categorical Data',value_name='Numerical Data')
        melted_category = melted_data[charts_columns_categorical[0]]
        for i in range(1,len(charts_columns_categorical)):
            melted_category = melted_category + '_' + melted_data[charts_columns_categorical[i]]
        melted_data['Categorical Data'] = melted_category
        plt.figure(figsize=(12,6))
        sns.kdeplot(melted_data,x='Numerical Data',hue='Categorical Data',common_norm = True,fill=True,palette='coolwarm')
        plt.xticks(rotation=70)
        with st.expander('Density Plot'):
            st.pyplot(plt.gcf())

  
    ## Funnel Chart
    if 'Funnel Chart' in user_select_charts:
        ## Melting Data for multiple categorical columns and multiple numerical columns
        melted_data = pd.melt(user_data_frame,id_vars=charts_columns_categorical,value_vars=charts_columns_numerical,var_name='Categorical Data',value_name='Numerical Data')
        melted_category = melted_data[charts_columns_categorical[0]]
        for i in range(1,len(charts_columns_categorical)):
            melted_category = melted_category + '_' + melted_data[charts_columns_categorical[i]]
        melted_data['Categorical Data'] = melted_category
        fig = px.funnel(melted_data,x='Categorical Data',y='Numerical Data')
        with st.expander('Funnel Chart'):
            st.plotly_chart(fig)

    ## Scatter Plot
    if 'Scatter Plot' in user_select_charts:
        # Melting Data for multiple categorical columns and multiple numerical columns
        melted_data = pd.melt(user_data_frame, id_vars=charts_columns_categorical, 
                            value_vars=charts_columns_numerical, 
                            var_name='Categorical Data', 
                            value_name='Numerical Data')

        # Combine categorical columns for x-axis
        melted_data['Categorical Data'] = melted_data[charts_columns_categorical[0]]
        for i in range(1, len(charts_columns_categorical)):
            melted_data['Categorical Data'] += '_' + melted_data[charts_columns_categorical[i]].astype(str)

        # Create the scatter plot
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=melted_data, x='Categorical Data', y='Numerical Data')
        plt.xticks(rotation=70)
        plt.title('Scatter Plot of Numerical Data')
        with st.expander('Scatter Plot'):
            st.pyplot(plt.gcf())

    ## Bubble Size
    if 'Bubble Chart' in user_select_charts:
        # Remove numerical columns that are not needed
        numerical_columns_charts = [col for col in numerical_columns if col not in charts_columns_numerical]
        
        # Select a column for bubble size
        bubble_size_col = st.selectbox('Enter the Column for creating Bubble Chart', options=numerical_columns_charts)
        
        if bubble_size_col:
            # Melting Data for multiple categorical columns and multiple numerical columns
            melted_data = pd.melt(user_data_frame, id_vars=charts_columns_categorical+[bubble_size_col], 
                                value_vars=charts_columns_numerical, 
                                var_name='Categorical Data', 
                                value_name='Numerical Data')

            # Combine categorical columns for x-axis
            melted_data['Categorical Data'] = melted_data[charts_columns_categorical[0]]
            for i in range(1, len(charts_columns_categorical)):
                melted_data['Categorical Data'] += '_' + melted_data[charts_columns_categorical[i]].astype(str)

            # Create the scatter plot
            plt.figure(figsize=(12, 6))
            sns.scatterplot(data=melted_data, x='Categorical Data', y='Numerical Data', 
                            size=melted_data[bubble_size_col], sizes=(20, 200), palette='viridis')
            plt.xticks(rotation=70)
            plt.title('Bubble Chart of Numerical Data')
            
            with st.expander('Bubble Chart'):
                st.pyplot(plt.gcf())
        else:
            st.write('Select a Column in Numerical Data')


    ## Counter plot
    if 'Count Plot' in user_select_charts:
        if len(chart_columns_select) == 1:
            plt.figure(figsize=(12,6))
            sns.countplot(data=user_data_frame,x=charts_columns_categorical[0],palette='viridis')
            with st.expander('Count Plot'):
                st.pyplot(plt.gcf())

    ## heat Map
    if 'Heat Map' in user_select_charts:
        
        heat_map_argfunc = st.selectbox('Choose a Argument Function :',options=['sum','count','mean'])
        ## Creating pivot table for creating heatmap for combining of both categorical and numerical columns
        pivot_table = user_data_frame.pivot_table(columns=charts_columns_categorical,values=charts_columns_numerical,aggfunc=heat_map_argfunc,fill_value=0)
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot_table,annot=True,cmap='YlGnBu', fmt='.1f')
        plt.title(f'Heat Map of Categorical and Numerical Data (Aggfunc: {heat_map_argfunc})')
        with st.expander('Heat Map'):
            st.pyplot(plt.gcf())
       
    ## Co-Relation Chart->> Finding Corealation between Numerical Variables
    if 'Co-relaton Chart(Heat Map)' in user_select_charts:
        plt.figure(figsize=(12,6))
        corr_matrix = user_data_frame[charts_columns_numerical].corr()
        sns.heatmap(corr_matrix,annot=True,cmap='viridis',square=True,fmt='.2f')
        plt.title('Co-relation Matrix')
        with st.expander('Co-relation Chart (Heat Map)'):
            st.pyplot(plt.gcf())

    ## Histogram
    if 'Histogram' in user_select_charts:
        plt.figure(figsize=(12,6))
        sns.histplot(data=user_data_frame,x=charts_columns_numerical[0],hue=charts_columns_categorical[0],bins=20,multiple='stack',alpha=0.3)
        with st.expander('Histogram'):
            st.pyplot(plt.gcf())



    st.subheader('Download the Csv File After Data Analysis')
    ### Downloading the DataFrame After Applying EXploratory Data Analysis
    def convert_csv(df):
        return df.to_csv(index=False).encode('utf-8')  # Use utf-8 encoding

    # Use the function and create a download button
    csv = convert_csv(user_data_frame)  # Ensure `data_frame` is defined earlier in your code

    st.download_button(
        label="Download Processed CSV File",
        data=csv,
        file_name=f'{uploaded_file.name}',  # Ensure `uploaded_file` is defined and has `.name` attribute
        mime='text/csv',
    )

    ## Feed Back
    text = st.text_area("How to Improve Website ? Give Suggestions :",max_chars=500)
    if len(text):
        text = ""
        st.balloons()



 

















