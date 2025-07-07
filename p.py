import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Energy Consumption Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #A23B72;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .energy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .daily-energy {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'daily_data' not in st.session_state:
    st.session_state.daily_data = []
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'current_day' not in st.session_state:
    st.session_state.current_day = 0

# Main title
st.markdown('<h1 class="main-header">⚡ Weekly Energy Consumption Tracker</h1>', unsafe_allow_html=True)

# Sidebar for user information
with st.sidebar:
    st.markdown('<h2 class="section-header">👤 User Information</h2>', unsafe_allow_html=True)
    
    name = st.text_input("Enter your name:", value=st.session_state.user_info.get('name', ''))
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=st.session_state.user_info.get('age', 25))
    city = st.text_input("Enter your city:", value=st.session_state.user_info.get('city', ''))
    area = st.text_input("Enter your area name:", value=st.session_state.user_info.get('area', ''))
    
    st.markdown('<h3 class="section-header">🏠 Housing Details</h3>', unsafe_allow_html=True)
    flat_tenament = st.selectbox("Housing Type:", ["Flat", "Tenament"], 
                                index=0 if st.session_state.user_info.get('flat_tenament') == 'Flat' else 1)
    facility = st.selectbox("BHK Type:", ["1BHK", "2BHK", "3BHK"],
                           index=["1BHK", "2BHK", "3BHK"].index(st.session_state.user_info.get('facility', '1BHK')))
    
    # Update session state
    st.session_state.user_info.update({
        'name': name, 'age': age, 'city': city, 'area': area,
        'flat_tenament': flat_tenament, 'facility': facility
    })

# Main content area
if name:  # Only show if name is entered
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Tab layout for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Daily Input", "📈 Weekly Summary", "💡 Energy Tips"])
    
    with tab1:
        st.markdown('<h2 class="section-header">Daily Energy Consumption Input</h2>', unsafe_allow_html=True)
        
        # Day selector
        selected_day = st.selectbox("Select Day:", days, index=st.session_state.current_day)
        day_index = days.index(selected_day)
        
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### Energy consumption for {selected_day}")
            
            # Base energy calculation
            base_energy = 0
            if facility == "1BHK":
                base_energy = 2 * 0.4 + 2 * 0.8  # 2 fans + 2 lights
                st.info("Base consumption (2 fans + 2 lights): 2.4 units")
            elif facility == "2BHK":
                base_energy = 3 * 0.4 + 3 * 0.8  # 3 fans + 3 lights
                st.info("Base consumption (3 fans + 3 lights): 3.6 units")
            elif facility == "3BHK":
                base_energy = 4 * 0.4 + 4 * 0.8  # 4 fans + 4 lights
                st.info("Base consumption (4 fans + 4 lights): 4.8 units")
            
            # Appliance usage
            st.markdown("#### Appliance Usage")
            ac = st.checkbox("Air Conditioner (3 units)", key=f"ac_{day_index}")
            fridge = st.checkbox("Refrigerator (3 units)", key=f"fridge_{day_index}")
            wm = st.checkbox("Washing Machine (3 units)", key=f"wm_{day_index}")
            
            # Calculate daily energy
            appliance_energy = 0
            if ac:
                appliance_energy += 3
            if fridge:
                appliance_energy += 3
            if wm:
                appliance_energy += 3
            
            total_daily_energy = base_energy + appliance_energy
            
            # Save button
            if st.button(f"Save {selected_day} Data", type="primary"):
                # Update or add daily data
                day_data = {
                    'day': selected_day,
                    'base_energy': base_energy,
                    'ac': ac,
                    'fridge': fridge,
                    'washing_machine': wm,
                    'appliance_energy': appliance_energy,
                    'total_energy': total_daily_energy
                }
                
                # Check if data for this day already exists
                existing_day = next((i for i, d in enumerate(st.session_state.daily_data) if d['day'] == selected_day), None)
                if existing_day is not None:
                    st.session_state.daily_data[existing_day] = day_data
                else:
                    st.session_state.daily_data.append(day_data)
                
                st.success(f"Data saved for {selected_day}!")
                st.rerun()
        
        with col2:
            st.markdown(f'<div class="energy-card"><h3>Daily Total</h3><h2>{total_daily_energy:.1f} units</h2></div>', unsafe_allow_html=True)
            
            # Energy breakdown
            st.markdown("#### Energy Breakdown")
            st.markdown(f"**Base:** {base_energy:.1f} units")
            st.markdown(f"**Appliances:** {appliance_energy:.1f} units")
            
            # Show saved days
            if st.session_state.daily_data:
                st.markdown("#### Saved Days")
                for day_data in st.session_state.daily_data:
                    st.markdown(f"✅ {day_data['day']}: {day_data['total_energy']:.1f} units")
    
    with tab2:
        st.markdown('<h2 class="section-header">Weekly Energy Summary</h2>', unsafe_allow_html=True)
        
        if st.session_state.daily_data:
            # Create DataFrame
            df = pd.DataFrame(st.session_state.daily_data)
            
            # Calculate weekly total
            weekly_total = df['total_energy'].sum()
            
            # Display summary cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Weekly Energy", f"{weekly_total:.1f} units")
            with col2:
                st.metric("Average Daily Energy", f"{weekly_total/len(df):.1f} units")
            with col3:
                st.metric("Days Recorded", len(df))
            with col4:
                estimated_monthly = weekly_total * 4.33
                st.metric("Est. Monthly Energy", f"{estimated_monthly:.0f} units")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily energy consumption chart
                fig_daily = px.bar(df, x='day', y='total_energy', 
                                 title='Daily Energy Consumption',
                                 color='total_energy',
                                 color_continuous_scale='viridis')
                fig_daily.update_layout(showlegend=False)
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                # Energy source breakdown
                appliance_cols = ['ac', 'fridge', 'washing_machine']
                appliance_names = ['Air Conditioner', 'Refrigerator', 'Washing Machine']
                
                appliance_usage = []
                for col, name in zip(appliance_cols, appliance_names):
                    count = df[col].sum()
                    appliance_usage.append({'Appliance': name, 'Days Used': count, 'Total Energy': count * 3})
                
                if appliance_usage:
                    appliance_df = pd.DataFrame(appliance_usage)
                    fig_pie = px.pie(appliance_df, values='Total Energy', names='Appliance', 
                                   title='Energy Distribution by Appliance')
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            # Weekly trend
            if len(df) > 1:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=df['day'], y=df['total_energy'], 
                                             mode='lines+markers', name='Energy Usage',
                                             line=dict(color='#2E86AB', width=3)))
                fig_trend.update_layout(title='Weekly Energy Trend', 
                                      xaxis_title='Day', yaxis_title='Energy (units)')
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Data table
            st.markdown("#### Detailed Daily Data")
            st.dataframe(df, use_container_width=True)
            
            # Export data
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name=f"{name}_energy_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No data recorded yet. Please use the 'Daily Input' tab to record your energy consumption.")
    
    with tab3:
        st.markdown('<h2 class="section-header">💡 Energy Saving Tips</h2>', unsafe_allow_html=True)
        
        tips = [
            "🌡️ Set your AC temperature to 24°C or higher to save energy",
            "💡 Switch to LED bulbs - they use 75% less energy than incandescent bulbs",
            "🔌 Unplug electronics when not in use to avoid phantom loads",
            "🌀 Use ceiling fans with AC to circulate air and feel cooler at higher temperatures",
            "🧊 Keep your refrigerator at 37-40°F and freezer at 0-5°F for optimal efficiency",
            "👕 Wash clothes in cold water when possible - heating water uses most energy",
            "🏠 Seal air leaks around windows and doors to improve insulation",
            "☀️ Use natural light during the day instead of artificial lighting",
            "🔋 Consider solar panels if your area gets good sunlight",
            "📱 Use smart power strips to automatically cut power to devices in standby mode"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
        
        # Energy efficiency calculator
        st.markdown("#### Energy Cost Calculator")
        cost_per_unit = st.number_input("Enter cost per unit (₹):", value=5.0, min_value=0.1)
        
        if st.session_state.daily_data:
            weekly_total = sum(d['total_energy'] for d in st.session_state.daily_data)
            weekly_cost = weekly_total * cost_per_unit
            monthly_cost = weekly_cost * 4.33
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weekly Cost", f"₹{weekly_cost:.2f}")
            with col2:
                st.metric("Monthly Cost", f"₹{monthly_cost:.2f}")

else:
    st.info("👋 Please enter your name in the sidebar to get started!")

# Footer
st.markdown("---")
st.markdown("*Energy Consumption Tracker - Built with Streamlit*")