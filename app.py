import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from PIL import Image

Im = Image.open('customer-retention-vector-icon-client-return-business-marketing-user-consumer-care-customer-retention-vector-icon-client-return-138279322.webp')
st.set_page_config(page_title= 'Customer Churn Prediction App',layout="wide", page_icon=Im)
with open('model_logic_churn.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

def assign_recency_score(recency, recency_quartiles):
    if recency <= recency_quartiles[0]:
        return 4
    elif recency <= recency_quartiles[1]:
        return 3
    elif recency <= recency_quartiles[2]:
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
    else:
        return 4

# Function to assign Monetary score based on predefined quartiles
def assign_monetary_score(monetary, monetary_quartiles):
    if monetary <= monetary_quartiles[0]:
        return 1
    elif monetary <= monetary_quartiles[1]:
        return 2
    elif monetary <= monetary_quartiles[2]:
        return 3
    else:
        return 4

# Function to assign customer segments based on RFM class
def Segment_macro(rfm_class):
    if rfm_class in ['112', '113', '122', '123', '133','132', '211', '212', '213', '221', '222', '223', '231','232', '233', '311', '312', '313', '321', '322', 
                     '323', '331', '332', '333', '411', '412', '413', '421','422', '423', '431','432', '433']:  
        return 'NewCustomer'
    elif rfm_class in ['114', '124', '134', '214', '224', '234', '314', '324', '334', '414', '424', '434']: 
        return 'Big Spender'
    elif rfm_class in ['121','131']:
        return 'Lost Customer'
    elif rfm_class in ['144','143', '142', '141', '242', '243', '244', '344', '343', '441', '442', '443']:
        return 'Loyal Customer'
    elif rfm_class in ['444']: 
        return 'Best Customer'
    elif rfm_class in ['111']:
        return 'DeadBeats'
    else:
        return 'Unknown'

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
    recency_quartiles = [17, 50, 141]
    frequency_quartiles = [1, 2, 5]
    monetary_quartiles = [307.245, 674.450, 1661.640]
    # Step 1: Calculate Recency (days since last purchase)
    recent_date = datetime(2011, 12, 9)
    last_purchase_date = pd.to_datetime(st.session_state.Last_purchase_date)
    recency = (recent_date - last_purchase_date).days

    # Step 2: Calculate Frequency (assuming 'frequency' is the number of purchases made)
    Frequency = st.session_state.frequency
    print(Frequency)

    # Step 3: Calculate Monetary (total purchase amount)
    Monetary = st.session_state.total_spent

    # Step 4: Assign Recency scores based on the quartiles
    r_quantile = assign_recency_score(recency, recency_quartiles)

    # Step 5: Assign Frequency scores based on the quartiles
    f_quantile = assign_frequency_score(Frequency, frequency_quartiles)

    # Step 6: Assign Monetary scores based on the quartiles
    m_quantile = assign_monetary_score(Monetary, monetary_quartiles)

    # Step 7: Combine RFM scores into a single RFM class
    rfm_quantile = str(r_quantile) + str(f_quantile) + str(m_quantile)

    # Step 8: Assign customer segment based on the RFM class
    customer_segment = Segment_macro(rfm_quantile)

    # Return the RFM class and segment for the customer
    return rfm_quantile, customer_segment

def display_input_fields(disabled=False):
    st.text_input('Customer Full Name *', value=st.session_state.get('Name', ''), disabled=disabled, key='Name_display')
    
    col1, col2 = st.columns(2)
    with col1:
        gender_options = ['Male', 'Female']
        gender_index = gender_options.index(st.session_state.get('Gender', 'Male'))
        st.radio('Gender *', gender_options, index=gender_index, horizontal=True, disabled=disabled, key='Gender_display')
    with col2:
        st.number_input('Age *', max_value=150, value=st.session_state.get('age', 1), disabled=disabled, key='age_display')
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input('Total Purchase Amount (Overall) *', min_value=1, value=st.session_state.get('total_spent', 1), disabled=disabled, key='total_spent_display')
    with col2:
        st.number_input('Unique Product Category *', min_value=1, value=st.session_state.get('purchase_diversity', 1), disabled=disabled, key='purchase_diversity_display')
    
    col1, col2 = st.columns(2)
    with col1:
        status_options = ['Canceled','Delivered', 'Returned']
        status_index = status_options.index(st.session_state.get('status', 'Canceled'))
        st.selectbox('Average Status of Product *', status_options, index=status_index, disabled=disabled, key='status_display')
    with col2:
        payment_options = ['COD',"EMI", "Paid via Wallet", "Paid via Banktransfer"]
        st.multiselect('Payment Method *', payment_options, default=st.session_state.get('payment_method', []), disabled=disabled, key='payment_method_display')

    col1, col2 = st.columns(2)
    with col1: 
        st.number_input('Average Quantity *', min_value=1,value=st.session_state.get('Quantity', 1), disabled=disabled, key='Quantity_display')
    with col2:
        st.number_input('Average Unit Price *', min_value=1,value=st.session_state.get('UnitPrice', 1), disabled=disabled, key='unitPrice_display')
    
    col1, col2 = st.columns(2)
    with col1:
        min_date = datetime(2010, 1, 1)
        max_date = datetime(2011, 12, 9)
        st.date_input('First Purchase Product Date', value=st.session_state.get('First_purchase_date', min_date), min_value=min_date, max_value=max_date, disabled=disabled, key='First_purchase_date_display')
    with col2:
        max_date = datetime(2011, 12, 9)
        st.date_input('Last Purchase Product Date', value=st.session_state.get('Last_purchase_date', max_date), max_value=max_date, disabled=disabled, key='Last_purchase_date_display')
    
    st.number_input('Frequency of Customer *', min_value=1, value=st.session_state.get('frequency', 1), disabled=disabled, key='frequency_display')
    
    col1, col2 = st.columns(2)
    with col1:
        st.number_input('Discount_Percent (%) *', min_value=0, value=st.session_state.get('Discount_Percent', 0), disabled=disabled, key='Discount_Percent_display')
    with col2:
        st.number_input('Return Rate *', min_value=0, value=st.session_state.get('return_rate', 0), disabled=disabled, key='return_rate_display')



def map_scalling_features(Quantity, UnitPrice,total_spent, recency, Frequency, Monetary, r_quantile,f_quantile,m_quantile,rfm_quantile):
    scaled_features = scaler.transform([[Quantity, UnitPrice,total_spent, recency, Frequency, Monetary, r_quantile,f_quantile,m_quantile,rfm_quantile]])
    return scaled_features[0]


def predict_output(scaled_features):
    # Create a single array of features for prediction
    features_for_prediction = [*scaled_features]
    prediction = model.predict([features_for_prediction])
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
                st.session_state.age = st.number_input('Age *', max_value=150, value=1)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.total_spent = st.number_input('Total Purchase Amount (Overall) *', min_value=1, value=1)
            with col2:
                st.session_state.purchase_diversity = st.number_input('Unique Product Category *', min_value=1)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.status = st.selectbox('Average Status of Product *', ['Canceled','Delivered', 'Returned'])
            with col2:
                st.session_state.payment_method = st.multiselect('Payment Method *', ['COD',"EMI", "Paid via Wallet", "Paid via Banktransfer"])
            # min_date = datetime(1980,1,1)
            # max_date = datetime(2021, 9, 30)
            # st.session_state.Customer_Since = st.date_input('Customer since',min_value=min_date, max_value=max_date)
            col1, col2 = st.columns(2)
            with col1: 
                st.session_state.Quantity = st.number_input('Average Quantity *', min_value=1)
            with col2:
                st.session_state.UnitPrice = st.number_input('Average Unit Price *', min_value=1)
            col1, col2 = st.columns(2)
            with col1:
                min_date = datetime(2010,1,1)
                max_date = datetime(2011, 12, 9)
                st.session_state.First_purchase_date = st.date_input('First Purchase Product Date',min_value=min_date, max_value=max_date)
            with col2:
                min_date = datetime(2010,1,1)
                max_date = datetime(2011, 12, 9)
                st.session_state.Last_purchase_date = st.date_input('Last Purchase Product Date',min_value=min_date, max_value=max_date)
            st.session_state.frequency = st.number_input('Frequency of Customer *', min_value=1)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.Discount_Percent = st.number_input('Discount_Percent (%) *', min_value=0)
            with col2:
                st.session_state.return_rate = st.number_input('Return Rate *', min_value=0)

            recency = (datetime(2021,9,30) - pd.to_datetime(st.session_state.Last_purchase_date)).days
            Frequency = st.session_state.frequency
            Monetary = st.session_state.total_spent
            recency_quartiles = [17, 50, 141]
            frequency_quartiles = [1, 2, 5]
            monetary_quartiles = [307.245, 674.450, 1661.640]
            r_quantile = assign_recency_score(recency, recency_quartiles)

            # Step 5: Assign Frequency scores based on the quartiles
            f_quantile = assign_frequency_score(Frequency, frequency_quartiles)

            # Step 6: Assign Monetary scores based on the quartiles
            m_quantile = assign_monetary_score(Monetary, monetary_quartiles)

            # Step 7: Combine RFM scores into a single RFM class
            rfm_quantile = str(r_quantile) + str(f_quantile) + str(m_quantile)

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
                scaled_features = map_scalling_features(st.session_state.Quantity,st.session_state.UnitPrice,st.session_state.total_spent, 
                                                        Monetary,recency,Frequency,r_quantile, f_quantile, m_quantile, rfm_quantile)
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
        

            