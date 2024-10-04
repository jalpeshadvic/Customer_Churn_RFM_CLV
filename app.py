import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime


with open('model_logic_churn.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def assign_recency_score(recency, recency_quartiles):
    if recency <= recency_quartiles[0]:
        return 5
    elif recency <= recency_quartiles[1]:
        return 4
    elif recency <= recency_quartiles[2]:
        return 3
    elif recency <= recency_quartiles[3]:
        return 2
    else:
        return 1

# Function to assign Frequency score based on predefined quartiles
def assign_frequency_score(frequency, frequency_quartiles):
    if frequency <= frequency_quartiles[0]:
        return 1
    elif frequency <= frequency_quartiles[1]:
        return 2
    elif frequency <= frequency_quartiles[2]:
        return 3
    elif frequency <= frequency_quartiles[3]:
        return 4
    else:
        return 5

# Function to assign Monetary score based on predefined quartiles
def assign_monetary_score(monetary, monetary_quartiles):
    if monetary <= monetary_quartiles[0]:
        return 1
    elif monetary <= monetary_quartiles[1]:
        return 2
    elif monetary <= monetary_quartiles[2]:
        return 3
    elif monetary <= monetary_quartiles[3]:
        return 4
    else:
        return 5

# Function to assign customer segments based on RFM class
def Segment_macro(rfm_class):
    if rfm_class in ['555', '554', '544', '545', '454', '455', '445', '543', '444', '435', '355', '354', '345', '344', '335',
                     '553', '552', '551', '541', '542', '533', '532', '531', '452', '451', '442', '441', '431', '453', '433', '432',
                     '423', '353', '352', '351', '342', '341', '333', '323']:
        return 'Loyal_Customer'
    elif rfm_class in ['512', '511', '422', '421', '412', '411', '311', '525', '524', '523', '522', '521', '515', '514', '513',
                       '425', '424', '413', '414', '415', '315', '314', '313', '535', '534', '443', '434', '343', '334', '325', '324']:
        return 'Promising_Customer'
    elif rfm_class in ['331', '321', '312', '221', '213', '231', '241', '251', '255', '254', '245', '244', '253', '252', '243', '242',
                       '235', '234', '225', '224', '152', '153', '145', '143', '142', '135', '134', '133', '125', '124', '155', '154',
                       '144', '214', '215', '115', '114', '113']:
        return 'Sleep_Customer'
    elif rfm_class in ['332', '322', '231', '241', '251', '233', '232', '223', '222', '132', '123', '122', '212', '211', '111', '112',
                       '121', '131', '141', '151']:
        return 'Lost_Customer'
    else:
        return 'Other'

def create_wizard_html(current_page):
    pages = ["Churn Prediction", "RFM Analysis", "Customer Lifetime Value"]
    html = """
    <style>
    .step-wizard {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .step {
        width: 30%;
        text-align: center;
        padding: 10px;
        background-color: #f0f0f0;
        border-radius: 5px;
        border: 2px solid #ccc;  /* Add this line for the border */
    }
    .active {
        background-color: #ADC2FF;
        font-weight: bold;
        border-color: #7B68EE;  /* Change border color for active step */
    }
    .completed {
        background-color: #90EE90;
        border-color: #2E8B57;  /* Change border color for completed step */
    }
    .completed-check {
        color: #4CAF50;
        margin-left: 5px;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    <div class="step-wizard">
    """
    for i, page in enumerate(pages):
        is_completed = st.session_state.get(f"{page.lower().replace(' ', '_')}_completed", False)
        checkmark = '<span class="completed-check">✔</span>' if is_completed else ''
        if page == current_page and not is_completed:
            html += f'<div class="step active">{page} {checkmark}</div>'
        elif is_completed:
            html += f'<div class="step completed">{page} {checkmark}</div>'
        else:
            html += f'<div class="step">{page}</div>'
    html += "</div>"
    return html

def rfm_analysis():
    recency_quartiles = [111, 164, 237, 280]
    frequency_quartiles = [1, 1, 3, 5]
    monetary_quartiles = [100, 260, 814.92, 3299.90]
    # Step 1: Calculate Recency (days since last purchase)
    recent_date = datetime(2021, 9, 30)
    last_purchase_date = pd.to_datetime(st.session_state.Last_purchase_date)
    recency = (recent_date - last_purchase_date).days

    # Step 2: Calculate Frequency (assuming 'frequency' is the number of purchases made)
    frequency = st.session_state.frequency

    # Step 3: Calculate Monetary (total purchase amount)
    monetary = st.session_state.total_spent

    # Step 4: Assign Recency scores based on the quartiles
    recency_score = assign_recency_score(recency, recency_quartiles)

    # Step 5: Assign Frequency scores based on the quartiles
    frequency_score = assign_frequency_score(frequency, frequency_quartiles)

    # Step 6: Assign Monetary scores based on the quartiles
    monetary_score = assign_monetary_score(monetary, monetary_quartiles)

    # Step 7: Combine RFM scores into a single RFM class
    rfm_class = str(recency_score) + str(frequency_score) + str(monetary_score)

    # Step 8: Assign customer segment based on the RFM class
    customer_segment = Segment_macro(rfm_class)

    # Return the RFM class and segment for the customer
    return rfm_class, customer_segment

def calculate_clv():
    # Ensure the Purchase_Date is in datetime format
    first_purchase = pd.to_datetime(st.session_state.First_purchase_date)
    last_purchase = pd.to_datetime(st.session_state.Last_purchase_date)
    
    # Calculate net revenue (assuming no returns for simplicity)
    net_revenue = st.session_state.total_spent

    order_count = st.session_state.frequency

    # Calculate Customer Lifespan (days between first and last purchase)
    customer_lifespan_days = (last_purchase - first_purchase).days

    # Convert Lifespan to years
    customer_lifespan_years = customer_lifespan_days / 365

    #frequency of customer lifespan
    ARPO = net_revenue / order_count

    # Calculate CLV
    clv = ARPO * order_count * customer_lifespan_years


    return clv

def map_categorical_inputs(status,category,payment_method,Gender):
    status_mapping = {'canceled' :0, 'closed':1, 'cod':2, 'complete':3, 'holded':4,
        'order_refunded':5, 'paid':6, 'pending':7, 'pending_paypal':8,
        'processing':9, 'received':10, 'refund':11}
    Product_Category_mapping = {'Appliances':0, 'Beauty & Grooming':1, 'Books':2, 'Computing':3,
        'Entertainment':4, 'Health & Sports':5, 'Home & Living':6, 'Kids & Baby':7,
        "Men's Fashion":8, 'Mobiles & Tablets':9, 'Others':10,
        'School & Education':11, 'Soghaat':12, 'Superstore':13, "Women's Fashion":14}
    Payment_Method_mapping = {'Easypay':0, 'Easypay_MA':1, 'Payaxis':2, 'apg':3, 'bankalfalah':4,
        'cashatdoorstep':5, 'cod':6, 'customercredit':7, 'easypay_voucher':8,
        'financesettlement':9, 'jazzvoucher':10, 'jazzwallet':11, 'mcblite':12}
    Gender_mapping = {'Female' :0 , 'Male' :1}

    mapped_status = status_mapping.get(status, -1)
    mapped_Product_Category = Product_Category_mapping.get(category,-1)
    mapped_Payment_Method = Payment_Method_mapping.get(payment_method,-1)
    mapped_Gender = Gender_mapping.get(Gender,-1)

    return mapped_status,mapped_Product_Category, mapped_Payment_Method, mapped_Gender

def display_input_fields(disabled=False):
    st.text_input('Customer Full Name *', value=st.session_state.get('Name', ''), disabled=disabled, key='Name_display')
    col1, col2 = st.columns(2)
    with col1:
        st.radio('Gender *', ['Male', 'Female'], horizontal=True, disabled=disabled, key='Gender_display', index=0 if st.session_state.get('Gender') == 'Male' else 1)
    with col2:
        st.number_input('Age *', min_value=1, max_value=80, value=st.session_state.get('age', 1), disabled=disabled, key='age_display')
    st.number_input('Total Purchase Amount *', min_value=1, value=st.session_state.get('total_spent', 1), disabled=disabled, key='total_spent_display')
    col1, col2 = st.columns(2)
    with col1:
        st.number_input('Quantity *', min_value=1, value=st.session_state.get('Quantity', 1), disabled=disabled, key='Quantity_display')
    with col2:
        st.number_input('Product Price *', min_value=1, value=st.session_state.get('Product_Price', 1), disabled=disabled, key='Product_Price_display')
    st.selectbox('Status *', ['canceled', 'closed', 'cod', 'complete', 'holded', 'order_refunded', 'paid', 'pending', 'pending_paypal', 'processing', 'received', 'refund'], disabled=disabled, key='status_display', index=['canceled', 'closed', 'cod', 'complete', 'holded', 'order_refunded', 'paid', 'pending', 'pending_paypal', 'processing', 'received', 'refund'].index(st.session_state.get('status', 'canceled')))
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox('Product Category *', ['Appliances', 'Beauty & Grooming', 'Books', 'Computing', 'Entertainment', 'Health & Sports', 'Home & Living', 'Kids & Baby', "Men's Fashion", 'Mobiles & Tablets', 'Others', 'School & Education', 'Soghaat', 'Superstore', "Women's Fashion"], disabled=disabled, key='category_display', index=['Appliances', 'Beauty & Grooming', 'Books', 'Computing', 'Entertainment', 'Health & Sports', 'Home & Living', 'Kids & Baby', "Men's Fashion", 'Mobiles & Tablets', 'Others', 'School & Education', 'Soghaat', 'Superstore', "Women's Fashion"].index(st.session_state.get('category', 'Appliances')))
    with col2:
        st.selectbox('Payment Method *', ['Easypay', 'Easypay_MA', 'Payaxis', 'apg', 'bankalfalah', 'cashatdoorstep', 'cod', 'customercredit', 'easypay_voucher', 'financesettlement', 'jazzvoucher', 'jazzwallet', 'mcblite'], disabled=disabled, key='payment_method_display', index=['Easypay', 'Easypay_MA', 'Payaxis', 'apg', 'bankalfalah', 'cashatdoorstep', 'cod', 'customercredit', 'easypay_voucher', 'financesettlement', 'jazzvoucher', 'jazzwallet', 'mcblite'].index(st.session_state.get('payment_method', 'Easypay')))
    st.date_input('Customer since', value=st.session_state.get('Customer_Since', datetime.now()), disabled=disabled, key='Customer_Since_display')
    col1, col2 = st.columns(2)
    with col1:
        st.date_input('First Purchase Date', value=st.session_state.get('First_purchase_date', datetime.now()), disabled=disabled, key='First_purchase_date_display')
    with col2:
        st.date_input('Last Purchase Date', value=st.session_state.get('Last_purchase_date', datetime.now()), disabled=disabled, key='Last_purchase_date_display')
    st.number_input('Frequency of Customer *', min_value=1, value=st.session_state.get('frequency', 1), disabled=disabled, key='frequency_display')
    st.number_input('Discount Amount *', min_value=0, value=st.session_state.get('discount_amount', 0), disabled=disabled, key='discount_amount_display')


def map_scalling_features(frequency, avg_order_value, monetary,recency,Discount_Percent,purchase_diversity,return_rate):
    scaled_features = scaler.transform([[frequency, avg_order_value, monetary,recency,Discount_Percent, purchase_diversity, return_rate]])
    return scaled_features[0]


def predict_output(scaled_features):
    # Create a single array of features for prediction
    features_for_prediction = [*scaled_features]
    prediction = model.predict_proba([features_for_prediction])[:, 1]
    return prediction

def main():
        
        def reset():
            st.session_state.page = "Churn Prediction"
            st.session_state.churn_prediction_completed = False
            st.session_state.rfm_analysis_completed = False
            st.session_state.customer_lifetime_value_completed = False
            st.session_state.prediction_result = None
            st.session_state.clv_result = None
            
            # Use st.rerun() instead of st.experimental_rerun()
            st.rerun()
            
        
        if "page" not in st.session_state:
            st.session_state.page = "Churn Prediction"
            st.session_state.churn_prediction_completed = False
            st.session_state.rfm_analysis_completed = False
            st.session_state.customer_lifetime_value_completed = False
            st.session_state.prediction_result = None
            st.session_state.clv_result = None
        
        html_temp = """
        <style>
            .sidebar .sidebar-content {
                background-color: #f0f0f0;
            }
            .sidebar .widget-radio .st-bb {
                padding-left: 40px;
                position: relative;
            }
            .sidebar .widget-radio .st-bb::before {
                content: '';
                position: absolute;
                left: 10px;
                top: 50%;
                transform: translateY(-50%);
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background-color: #ccc;
            }
            .sidebar .widget-radio .st-bb.completed::before {
                background-color: #90EE90;
            }
            .sidebar .widget-radio .st-bb.active::before {
                background-color: #ADC2FF;
            }
            .completed-check {
            color: #4CAF50;
            margin-left: 5px;
            font-size: 20px;
            font-weight: bold;
            }
        </style>
        <div style="background-color: #ADC2FF; padding: 1px; margin-bottom: 10px;">
            <h2 style="color: black; text-align:center;">Customer Churn Prediction</h2>
        </div>
        <div style="text-align: right; margin-bottom: 20px;">
            <p style="color: red; font-style: italic;">All fields are compulsory.</p>
        </div>
        """
        st.markdown(html_temp, unsafe_allow_html=True)

        pages = ["Churn Prediction", "RFM Analysis", "Customer Lifetime Value", "EDA for Customer"]

        st.markdown(create_wizard_html(st.session_state.page), unsafe_allow_html=True)

        st.sidebar.markdown("<span style='font-size: 20px; font-weight: bold; color: #520052;'>Customer Details</span>", unsafe_allow_html=True)
        for page in pages:
            is_completed = st.session_state.get(f"{page.lower().replace(' ', '_')}_completed", False)
            is_current = st.session_state.page == page
            checkmark = '✔' if is_completed else ''
            color = '#90EE90' if is_completed else ('#ADC2FF' if is_current else '#f0f0f0')
            st.sidebar.markdown(f'<div style="background-color: {color}; padding: 10px; border-radius: 5px; margin-bottom: 5px;">{page} <span style="color: green;">{checkmark}</span></div>', unsafe_allow_html=True)
        # Create a list to store the classes for each page
        sidebar_classes = ["", "", ""]

        # Set the appropriate class based on the current page
        current_index = pages.index(st.session_state.page)
        for i in range(len(pages)):
            if i < current_index:
                sidebar_classes[i] = "completed"
            elif i == current_index:
                sidebar_classes[i] = "active"
        
        if st.session_state.page == "Churn Prediction":

            st.session_state.Name = st.text_input('Customer Full Name *')
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.Gender = st.radio('Gender *',['Male','Female'],horizontal=True)
            with col2:
                st.session_state.age = st.number_input('Age *', min_value=1, max_value=80, value=1)
            st.session_state.total_spent = st.number_input('Total Purchase Amount *', min_value=1, value=1)
            # col1, col2 = st.columns(2)
            # with col1:
            #     st.session_state.Quantity = st.number_input('Quantity *', min_value=1, value=1)
            # with col2:
            #     st.session_state.Product_Price = st.number_input('Product Price *', min_value=1, value=1)
            st.session_state.status = st.selectbox('Status *', ['canceled', 'closed', 'cod', 'complete', 'holded',
                                                            'order_refunded', 'paid', 'pending', 'pending_paypal',
                                                            'processing', 'received', 'refund'])
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.purchase_diversity = st.number_input('Product Category *', min_value=1)
            with col2:
                st.session_state.payment_method = st.selectbox('Payment Method *', ['Easypay', 'Easypay_MA', 'Payaxis', 'apg', 'bankalfalah',
                                                                            'cashatdoorstep', 'cod', 'customercredit', 'easypay_voucher',
                                                                            'financesettlement', 'jazzvoucher', 'jazzwallet', 'mcblite'])
            min_date = datetime(1980,1,1)
            max_date = datetime(2021, 9, 30)
            st.session_state.Customer_Since = st.date_input('Customer since',min_value=min_date, max_value=max_date)
            col1, col2 = st.columns(2)
            with col1:
                min_date = datetime(1980,1,1)
                max_date = datetime(2021, 9, 30)
                st.session_state.First_purchase_date = st.date_input('First Purchase Date',min_value=min_date, max_value=max_date)
            with col2:
                max_date = datetime(2021, 9, 30)
                st.session_state.Last_purchase_date = st.date_input('Last Purchase Date', max_value=max_date)
            st.session_state.frequency = st.number_input('Frequency of Customer *', min_value=1)
            st.session_state.Discount_Percent = st.number_input('Discount_Percent *', min_value=0)
            st.session_state.return_rate = st.number_input('Return Rate *', min_value=0)

            recency = (datetime(2021,9,30) - pd.to_datetime(st.session_state.Last_purchase_date)).days
            frequency = st.session_state.frequency
            monetary = st.session_state.total_spent
            avg_order_value = monetary / frequency

            # mapped_status, mapped_Product_Category,mapped_Payment_Method,mapped_Gender = map_categorical_inputs(st.session_state.status,st.session_state.category, 
            #                                                                                     st.session_state.payment_method, st.session_state.Gender)

            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                col_predict, col_next = st.columns(2)
                with col_predict:
                    predict_button = st.button("Predict", key="predict_button")
                with col_next:
                    next_button = st.button("Next", key="next_button")
                
            if predict_button:
                scaled_features = map_scalling_features(frequency, avg_order_value, monetary,recency,st.session_state.Discount_Percent, st.session_state.purchase_diversity, st.session_state.return_rate)
                # map_categorical_inputs(st.session_state.Product_Category, st.session_state.Payment_Method, st.session_state.Gender)
                result = predict_output(scaled_features)
                formatted_result = "{:.2f}".format(float(result[0]))
                st.session_state.prediction_result = formatted_result
                st.success(f"Probability of Customer Churn is {st.session_state.prediction_result}")
            if next_button:
                st.session_state.churn_prediction_completed = True
                st.session_state.page = "RFM Analysis"
                st.rerun()

        elif st.session_state.page == "RFM Analysis":

            display_input_fields(disabled=True)


            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                col_rfm, col_next = st.columns(2)
            with col_rfm:
                rfm_button = st.button("Predict", key="rfm_button")
            with col_next:
                next_button = st.button("Next", key="next_button")
        
            if rfm_button:
                # Perform RFM analysis based on the user inputs
                    rfm_class, customer_segment = rfm_analysis()       
                # Display the RFM result and customer segment
                    st.success(f"Customer {st.session_state.Name}'s RFM Class is {rfm_class} and Segment is {customer_segment}")
            if next_button:
                st.session_state.rfm_analysis_completed = True
                st.session_state.page = "Customer Lifetime Value"
                st.rerun()

        elif st.session_state.page == "Customer Lifetime Value":

            display_input_fields(disabled=True)

            col1, col2, col3 = st.columns([1,2,1])

            with col2:
                col_clv, col_reset = st.columns(2)
                with col_clv:
                    clv_button = st.button("Predict", key="clv_button")
                with col_reset:
                    reset_button = st.button("Reset", key="reset_button")

            if clv_button:
                
                result = calculate_clv()
                clv_result_score = "{:.2f}".format(result)
                st.session_state.clv_result = clv_result_score
                st.session_state.customer_lifetime_value_completed = True
                st.rerun() 
            
            if st.session_state.clv_result:
                st.success(f"Customer {st.session_state.Name}'s Lifetime Value (CLV) is {st.session_state.clv_result}")
                

            if reset_button:
                reset()


if __name__ == '__main__':
    main()
        

            