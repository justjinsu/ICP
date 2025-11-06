import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Carbon Pricing Calculator",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E7D32;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976D2;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F57C00;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_calc' not in st.session_state:
    st.session_state.df_calc = None

# Title and description
st.markdown('<p class="main-header">🌍 Internal Carbon Pricing Calculator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced MACC Analysis and Carbon Pricing Metrics Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/2E7D32/FFFFFF?text=Carbon+Pricing", use_container_width=True)

    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload MACC Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing your MACC data"
    )

    st.markdown("---")

    # Configuration section
    st.header("⚙️ Configuration")

    # Currency selection
    currency = st.selectbox(
        "Currency",
        options=["USD ($)", "EUR (€)", "GBP (£)", "JPY (¥)", "CNY (¥)"],
        index=0
    )
    currency_symbol = currency.split("(")[1].strip(")")

    # Unit selection
    co2_unit = st.selectbox(
        "CO2 Unit",
        options=["tCO2e", "ktCO2e", "MtCO2e"],
        index=0
    )

    st.markdown("---")

    # Calculation methodology
    st.header("📊 Methodology")

    calculation_method = st.radio(
        "ICP Calculation Method",
        options=["Weighted Average", "Simple Average", "Marginal Cost"],
        help="Choose how to calculate the Internal Carbon Price"
    )

    include_negative_costs = st.checkbox(
        "Include Negative Costs",
        value=True,
        help="Include measures with negative costs (cost savings)"
    )

    # Filtering options
    st.markdown("---")
    st.header("🔍 Filters")

    enable_cost_filter = st.checkbox("Enable Cost Filtering", value=False)
    if enable_cost_filter:
        cost_range = st.slider(
            "Cost Range",
            min_value=-100.0,
            max_value=200.0,
            value=(-100.0, 200.0),
            step=5.0
        )

# Main content
if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

        df = st.session_state.df

        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📋 Data Overview",
            "💰 Carbon Pricing Metrics",
            "📈 MACC Analysis",
            "🎯 Scenario Analysis",
            "📊 Statistical Analysis",
            "📥 Export & Reports"
        ])

        # Tab 1: Data Overview
        with tab1:
            st.header("Data Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Duplicate Rows", df.duplicated().sum())

            st.subheader("Column Mapping")

            col_map1, col_map2, col_map3 = st.columns(3)

            with col_map1:
                abatement_col = st.selectbox(
                    f"Abatement Potential Column ({co2_unit})",
                    options=df.columns.tolist(),
                    index=0,
                    key="abatement_col"
                )

            with col_map2:
                cost_col = st.selectbox(
                    f"Cost per Unit Column ({currency_symbol}/{co2_unit})",
                    options=df.columns.tolist(),
                    index=1 if len(df.columns) > 1 else 0,
                    key="cost_col"
                )

            with col_map3:
                measure_col = st.selectbox(
                    "Measure/Project Name Column (Optional)",
                    options=["None"] + df.columns.tolist(),
                    index=0,
                    key="measure_col"
                )

            # Additional optional columns
            with st.expander("Advanced Column Mapping"):
                col_adv1, col_adv2 = st.columns(2)
                with col_adv1:
                    year_col = st.selectbox(
                        "Year/Time Period (Optional)",
                        options=["None"] + df.columns.tolist(),
                        index=0
                    )
                with col_adv2:
                    category_col = st.selectbox(
                        "Category/Sector (Optional)",
                        options=["None"] + df.columns.tolist(),
                        index=0
                    )

            # Data preprocessing
            st.subheader("Data Preprocessing")

            try:
                df_calc = df[[abatement_col, cost_col]].copy()
                df_calc.columns = ['abatement', 'cost']

                if measure_col != "None":
                    df_calc['measure'] = df[measure_col]
                else:
                    df_calc['measure'] = [f"Measure {i+1}" for i in range(len(df_calc))]

                # Show data quality issues
                issues = []
                null_count = df_calc[['abatement', 'cost']].isnull().sum().sum()
                if null_count > 0:
                    issues.append(f"⚠️ {null_count} missing values detected")

                non_numeric_abatement = df_calc['abatement'].apply(lambda x: not isinstance(x, (int, float))).sum()
                non_numeric_cost = df_calc['cost'].apply(lambda x: not isinstance(x, (int, float))).sum()

                if non_numeric_abatement > 0:
                    issues.append(f"⚠️ {non_numeric_abatement} non-numeric values in abatement column")
                if non_numeric_cost > 0:
                    issues.append(f"⚠️ {non_numeric_cost} non-numeric values in cost column")

                if issues:
                    st.warning("Data Quality Issues Found:")
                    for issue in issues:
                        st.markdown(issue)

                # Convert to numeric
                df_calc['abatement'] = pd.to_numeric(df_calc['abatement'], errors='coerce')
                df_calc['cost'] = pd.to_numeric(df_calc['cost'], errors='coerce')

                # Drop missing values
                rows_before = len(df_calc)
                df_calc = df_calc.dropna(subset=['abatement', 'cost'])
                rows_after = len(df_calc)

                if rows_before != rows_after:
                    st.info(f"Removed {rows_before - rows_after} rows with missing or invalid data")

                # Apply filters
                if not include_negative_costs:
                    df_calc = df_calc[df_calc['cost'] >= 0]

                if enable_cost_filter:
                    df_calc = df_calc[(df_calc['cost'] >= cost_range[0]) & (df_calc['cost'] <= cost_range[1])]

                # Sort by cost
                df_calc = df_calc.sort_values('cost').reset_index(drop=True)
                df_calc['cumulative_abatement'] = df_calc['abatement'].cumsum()

                st.session_state.df_calc = df_calc

                st.success(f"✅ Data processed successfully! {len(df_calc)} measures ready for analysis.")

            except Exception as e:
                st.error(f"Error processing data: {str(e)}")
                st.stop()

            # Display processed data
            st.subheader("Processed Data")

            display_df = df_calc.copy()
            display_df['abatement'] = display_df['abatement'].round(2)
            display_df['cost'] = display_df['cost'].round(2)
            display_df['cumulative_abatement'] = display_df['cumulative_abatement'].round(2)

            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            # Basic statistics
            st.subheader("Basic Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Abatement Potential**")
                abatement_stats = df_calc['abatement'].describe()
                st.dataframe(abatement_stats, use_container_width=True)

            with col2:
                st.markdown("**Cost per Unit**")
                cost_stats = df_calc['cost'].describe()
                st.dataframe(cost_stats, use_container_width=True)

        # Tab 2: Carbon Pricing Metrics
        with tab2:
            st.header("Carbon Pricing Metrics")

            if st.session_state.df_calc is not None:
                df_calc = st.session_state.df_calc

                # Check if we have data
                if len(df_calc) == 0:
                    st.error("No data available after filtering. Please adjust your filters or upload different data.")
                    st.stop()

                # Calculate all metrics
                total_abatement = df_calc['abatement'].sum()
                total_cost = (df_calc['abatement'] * df_calc['cost']).sum()

                # Internal Carbon Price (ICP)
                if calculation_method == "Weighted Average":
                    icp = total_cost / total_abatement if total_abatement > 0 else 0
                    icp_description = "Weighted average cost across all abatement measures"
                elif calculation_method == "Simple Average":
                    icp = df_calc['cost'].mean() if len(df_calc) > 0 else 0
                    icp_description = "Simple average of all measure costs"
                else:  # Marginal Cost
                    icp = df_calc['cost'].iloc[-1] if len(df_calc) > 0 else 0
                    icp_description = "Marginal cost of the last abatement measure"

                # Shadow Price of Carbon (SPC)
                spc = df_calc['cost'].iloc[-1] if len(df_calc) > 0 else 0

                # Implicit Carbon Price
                implicit_price = df_calc['cost'].median() if len(df_calc) > 0 else 0

                # Additional metrics
                mean_cost = df_calc['cost'].mean() if len(df_calc) > 0 else 0
                std_cost = df_calc['cost'].std() if len(df_calc) > 1 else 0

                # Display main metrics
                st.subheader("Primary Carbon Pricing Indicators")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Internal Carbon Price (ICP)",
                        value=f"{currency_symbol}{icp:.2f}/{co2_unit}",
                        help=icp_description
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Shadow Price of Carbon (SPC)",
                        value=f"{currency_symbol}{spc:.2f}/{co2_unit}",
                        help="Marginal cost of the last abatement measure implemented"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="Implicit Carbon Price",
                        value=f"{currency_symbol}{implicit_price:.2f}/{co2_unit}",
                        help="Median cost across all abatement measures"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("---")

                # Secondary metrics
                st.subheader("Secondary Indicators")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Mean Cost", f"{currency_symbol}{mean_cost:.2f}/{co2_unit}")
                with col2:
                    st.metric("Std Deviation", f"{currency_symbol}{std_cost:.2f}/{co2_unit}")
                with col3:
                    st.metric("Min Cost", f"{currency_symbol}{df_calc['cost'].min():.2f}/{co2_unit}")
                with col4:
                    st.metric("Max Cost", f"{currency_symbol}{df_calc['cost'].max():.2f}/{co2_unit}")

                st.markdown("---")

                # Abatement summary
                st.subheader("Abatement Summary")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Abatement Potential", f"{total_abatement:,.0f} {co2_unit}")
                with col2:
                    st.metric("Number of Measures", len(df_calc))
                with col3:
                    total_cost_investment = (df_calc['abatement'] * df_calc['cost']).sum()
                    st.metric("Total Investment Required", f"{currency_symbol}{total_cost_investment:,.0f}")
                with col4:
                    negative_cost_measures = len(df_calc[df_calc['cost'] < 0])
                    st.metric("Negative Cost Measures", negative_cost_measures)

                # Cost distribution visualization
                st.subheader("Cost Distribution Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=df_calc['cost'],
                        nbinsx=20,
                        marker_color='rgba(46, 125, 50, 0.7)',
                        name='Cost Distribution'
                    ))
                    fig_hist.update_layout(
                        title="Cost Distribution",
                        xaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    # Box plot
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=df_calc['cost'],
                        marker_color='rgba(46, 125, 50, 0.7)',
                        name='Cost'
                    ))
                    fig_box.update_layout(
                        title="Cost Box Plot",
                        yaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True)

                # Quartile analysis
                st.subheader("Quartile Analysis")

                quartiles = df_calc['cost'].quantile([0.25, 0.5, 0.75])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("1st Quartile (Q1)", f"{currency_symbol}{quartiles[0.25]:.2f}/{co2_unit}")
                with col2:
                    st.metric("2nd Quartile (Q2/Median)", f"{currency_symbol}{quartiles[0.5]:.2f}/{co2_unit}")
                with col3:
                    st.metric("3rd Quartile (Q3)", f"{currency_symbol}{quartiles[0.75]:.2f}/{co2_unit}")

                # Top and bottom measures
                st.subheader("Measure Insights")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top 5 Most Cost-Effective Measures**")
                    top_5 = df_calc.nsmallest(5, 'cost')[['measure', 'cost', 'abatement']]
                    top_5_display = top_5.copy()
                    top_5_display.columns = ['Measure', f'Cost ({currency_symbol}/{co2_unit})', f'Abatement ({co2_unit})']
                    st.dataframe(top_5_display, use_container_width=True, hide_index=True)

                with col2:
                    st.markdown("**Top 5 Most Expensive Measures**")
                    bottom_5 = df_calc.nlargest(5, 'cost')[['measure', 'cost', 'abatement']]
                    bottom_5_display = bottom_5.copy()
                    bottom_5_display.columns = ['Measure', f'Cost ({currency_symbol}/{co2_unit})', f'Abatement ({co2_unit})']
                    st.dataframe(bottom_5_display, use_container_width=True, hide_index=True)

        # Tab 3: MACC Analysis
        with tab3:
            st.header("Marginal Abatement Cost Curve (MACC) Analysis")

            if st.session_state.df_calc is not None:
                df_calc = st.session_state.df_calc

                if len(df_calc) == 0:
                    st.error("No data available after filtering. Please adjust your filters or upload different data.")
                    st.stop()

                # MACC visualization options
                col1, col2, col3 = st.columns(3)

                with col1:
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        options=["Cost-based", "Uniform", "Alternating", "Category-based"],
                        index=0
                    )

                with col2:
                    show_labels = st.checkbox("Show Measure Labels", value=False)

                with col3:
                    show_reference_lines = st.checkbox("Show Reference Lines", value=True)

                # Create MACC curve
                fig_macc = go.Figure()

                # Add bars for each measure
                for i in range(len(df_calc)):
                    x_start = df_calc['cumulative_abatement'].iloc[i-1] if i > 0 else 0
                    x_end = df_calc['cumulative_abatement'].iloc[i]
                    y = df_calc['cost'].iloc[i]
                    measure_name = df_calc['measure'].iloc[i]
                    abatement = df_calc['abatement'].iloc[i]

                    # Determine color
                    if color_scheme == "Cost-based":
                        if y < 0:
                            color = 'rgba(46, 125, 50, 0.7)'  # Green for negative cost
                        elif y < implicit_price:
                            color = 'rgba(255, 193, 7, 0.7)'  # Yellow for below median
                        else:
                            color = 'rgba(211, 47, 47, 0.7)'  # Red for above median
                    elif color_scheme == "Uniform":
                        color = 'rgba(100, 150, 200, 0.7)'
                    elif color_scheme == "Alternating":
                        color = 'rgba(100, 150, 200, 0.7)' if i % 2 == 0 else 'rgba(150, 100, 200, 0.7)'
                    else:  # Category-based would need category data
                        color = 'rgba(100, 150, 200, 0.7)'

                    fig_macc.add_trace(go.Scatter(
                        x=[x_start, x_end, x_end, x_start, x_start],
                        y=[0, 0, y, y, 0],
                        fill='toself',
                        fillcolor=color,
                        line=dict(color=color.replace('0.7', '1.0'), width=2),
                        hovertemplate=f'<b>{measure_name}</b><br>' +
                                    f'Abatement: {abatement:.0f} {co2_unit}<br>' +
                                    f'Cost: {currency_symbol}{y:.2f}/{co2_unit}<br>' +
                                    f'Total Cost: {currency_symbol}{(abatement * y):,.0f}<extra></extra>',
                        showlegend=False,
                        name=measure_name
                    ))

                    # Add labels if requested
                    if show_labels and len(df_calc) <= 20:  # Only show labels for reasonable number of measures
                        fig_macc.add_annotation(
                            x=(x_start + x_end) / 2,
                            y=y,
                            text=measure_name,
                            showarrow=False,
                            font=dict(size=8),
                            textangle=-90 if y > 0 else 90
                        )

                # Add reference lines
                if show_reference_lines:
                    # ICP line
                    fig_macc.add_hline(
                        y=icp,
                        line_dash="dash",
                        line_color="blue",
                        annotation_text=f"ICP: {currency_symbol}{icp:.2f}",
                        annotation_position="right"
                    )

                    # Implicit price line
                    fig_macc.add_hline(
                        y=implicit_price,
                        line_dash="dot",
                        line_color="purple",
                        annotation_text=f"Implicit: {currency_symbol}{implicit_price:.2f}",
                        annotation_position="right"
                    )

                    # Zero line
                    fig_macc.add_hline(
                        y=0,
                        line_color="black",
                        line_width=1
                    )

                fig_macc.update_layout(
                    title="Marginal Abatement Cost Curve (MACC)",
                    xaxis_title=f"Cumulative Abatement Potential ({co2_unit})",
                    yaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                    hovermode='closest',
                    height=600,
                    showlegend=False
                )

                st.plotly_chart(fig_macc, use_container_width=True)

                # Cost curve analysis
                st.subheader("Cost Curve Breakdown")

                col1, col2, col3 = st.columns(3)

                negative_cost = df_calc[df_calc['cost'] < 0]
                low_cost = df_calc[(df_calc['cost'] >= 0) & (df_calc['cost'] < implicit_price)]
                high_cost = df_calc[df_calc['cost'] >= implicit_price]

                with col1:
                    st.markdown("**Negative Cost Measures**")
                    st.metric("Count", len(negative_cost))
                    st.metric("Total Abatement", f"{negative_cost['abatement'].sum():,.0f} {co2_unit}")
                    st.metric("Net Savings", f"{currency_symbol}{abs((negative_cost['abatement'] * negative_cost['cost']).sum()):,.0f}")

                with col2:
                    st.markdown("**Low Cost Measures**")
                    st.markdown(f"*(Below median: {currency_symbol}{implicit_price:.2f})*")
                    st.metric("Count", len(low_cost))
                    st.metric("Total Abatement", f"{low_cost['abatement'].sum():,.0f} {co2_unit}")
                    st.metric("Total Cost", f"{currency_symbol}{(low_cost['abatement'] * low_cost['cost']).sum():,.0f}")

                with col3:
                    st.markdown("**High Cost Measures**")
                    st.markdown(f"*(Above median: {currency_symbol}{implicit_price:.2f})*")
                    st.metric("Count", len(high_cost))
                    st.metric("Total Abatement", f"{high_cost['abatement'].sum():,.0f} {co2_unit}")
                    st.metric("Total Cost", f"{currency_symbol}{(high_cost['abatement'] * high_cost['cost']).sum():,.0f}")

                # Cumulative cost curve
                st.subheader("Cumulative Cost Analysis")

                df_calc['cumulative_cost'] = (df_calc['abatement'] * df_calc['cost']).cumsum()

                fig_cum = go.Figure()

                fig_cum.add_trace(go.Scatter(
                    x=df_calc['cumulative_abatement'],
                    y=df_calc['cumulative_cost'],
                    mode='lines+markers',
                    line=dict(color='rgba(46, 125, 50, 0.8)', width=3),
                    marker=dict(size=8),
                    name='Cumulative Cost'
                ))

                fig_cum.update_layout(
                    title="Cumulative Investment vs. Cumulative Abatement",
                    xaxis_title=f"Cumulative Abatement ({co2_unit})",
                    yaxis_title=f"Cumulative Investment ({currency_symbol})",
                    height=400
                )

                st.plotly_chart(fig_cum, use_container_width=True)

        # Tab 4: Scenario Analysis
        with tab4:
            st.header("Scenario Analysis")

            if st.session_state.df_calc is not None:
                df_calc = st.session_state.df_calc

                if len(df_calc) == 0:
                    st.error("No data available after filtering. Please adjust your filters or upload different data.")
                    st.stop()

                st.markdown("""
                Analyze different scenarios by setting carbon price targets or abatement goals.
                This helps identify which measures are cost-effective under different carbon pricing regimes.
                """)

                # Scenario type selection
                scenario_type = st.radio(
                    "Scenario Type",
                    options=["Carbon Price Target", "Abatement Target"],
                    horizontal=True
                )

                if scenario_type == "Carbon Price Target":
                    st.subheader("Carbon Price Target Analysis")

                    target_price = st.slider(
                        f"Target Carbon Price ({currency_symbol}/{co2_unit})",
                        min_value=float(df_calc['cost'].min()),
                        max_value=float(df_calc['cost'].max()),
                        value=float(implicit_price),
                        step=1.0
                    )

                    # Filter measures below target price
                    viable_measures = df_calc[df_calc['cost'] <= target_price]

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        viable_pct = (len(viable_measures)/len(df_calc)*100) if len(df_calc) > 0 else 0
                        st.metric(
                            "Viable Measures",
                            len(viable_measures),
                            f"{viable_pct:.1f}% of total"
                        )

                    with col2:
                        viable_abatement = viable_measures['abatement'].sum()
                        total_abatement = df_calc['abatement'].sum()
                        viable_abatement_pct = (viable_abatement/total_abatement*100) if total_abatement > 0 else 0
                        st.metric(
                            "Total Abatement",
                            f"{viable_abatement:,.0f} {co2_unit}",
                            f"{viable_abatement_pct:.1f}% of total"
                        )

                    with col3:
                        viable_cost = (viable_measures['abatement'] * viable_measures['cost']).sum()
                        st.metric(
                            "Total Investment",
                            f"{currency_symbol}{viable_cost:,.0f}"
                        )

                    with col4:
                        avg_cost = viable_cost / viable_abatement if viable_abatement > 0 else 0
                        st.metric(
                            "Average Cost",
                            f"{currency_symbol}{avg_cost:.2f}/{co2_unit}"
                        )

                    # Visualization
                    fig_scenario = go.Figure()

                    # Add all measures
                    for i in range(len(df_calc)):
                        x_start = df_calc['cumulative_abatement'].iloc[i-1] if i > 0 else 0
                        x_end = df_calc['cumulative_abatement'].iloc[i]
                        y = df_calc['cost'].iloc[i]

                        # Color based on viability
                        if y <= target_price:
                            color = 'rgba(46, 125, 50, 0.7)'  # Green - viable
                        else:
                            color = 'rgba(189, 189, 189, 0.3)'  # Gray - not viable

                        fig_scenario.add_trace(go.Scatter(
                            x=[x_start, x_end, x_end, x_start, x_start],
                            y=[0, 0, y, y, 0],
                            fill='toself',
                            fillcolor=color,
                            line=dict(color=color.replace('0.7', '1.0').replace('0.3', '0.8'), width=1),
                            showlegend=False,
                            hovertemplate=f'Cost: {currency_symbol}{y:.2f}/{co2_unit}<br>' +
                                        f'Status: {"Viable" if y <= target_price else "Not Viable"}<extra></extra>'
                        ))

                    # Add target price line
                    fig_scenario.add_hline(
                        y=target_price,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"Target Price: {currency_symbol}{target_price:.2f}",
                        annotation_position="right"
                    )

                    fig_scenario.update_layout(
                        title=f"Viable Measures at {currency_symbol}{target_price:.2f}/{co2_unit}",
                        xaxis_title=f"Cumulative Abatement Potential ({co2_unit})",
                        yaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                        height=500
                    )

                    st.plotly_chart(fig_scenario, use_container_width=True)

                    # List of viable measures
                    if len(viable_measures) > 0:
                        st.subheader("Viable Measures")
                        viable_display = viable_measures[['measure', 'cost', 'abatement']].copy()
                        viable_display['total_cost'] = viable_display['cost'] * viable_display['abatement']
                        viable_display.columns = ['Measure', f'Cost ({currency_symbol}/{co2_unit})', f'Abatement ({co2_unit})', f'Total Cost ({currency_symbol})']
                        st.dataframe(viable_display, use_container_width=True, hide_index=True)

                else:  # Abatement Target
                    st.subheader("Abatement Target Analysis")

                    max_abatement = df_calc['abatement'].sum()

                    target_abatement = st.slider(
                        f"Target Abatement ({co2_unit})",
                        min_value=0.0,
                        max_value=float(max_abatement),
                        value=float(max_abatement * 0.5),
                        step=float(max_abatement * 0.05)
                    )

                    # Find measures needed to reach target
                    df_sorted = df_calc.sort_values('cost')
                    cumsum = df_sorted['abatement'].cumsum()
                    needed_measures = df_sorted[cumsum <= target_abatement]

                    # If we haven't reached target, add one more measure
                    if len(needed_measures) < len(df_sorted) and cumsum.iloc[-1] >= target_abatement:
                        next_idx = len(needed_measures)
                        if next_idx < len(df_sorted):
                            needed_measures = pd.concat([needed_measures, df_sorted.iloc[[next_idx]]])

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        needed_pct = (len(needed_measures)/len(df_calc)*100) if len(df_calc) > 0 else 0
                        st.metric(
                            "Measures Required",
                            len(needed_measures),
                            f"{needed_pct:.1f}% of total"
                        )

                    with col2:
                        actual_abatement = needed_measures['abatement'].sum()
                        st.metric(
                            "Actual Abatement",
                            f"{actual_abatement:,.0f} {co2_unit}",
                            f"{(actual_abatement-target_abatement):,.0f} {co2_unit}"
                        )

                    with col3:
                        total_investment = (needed_measures['abatement'] * needed_measures['cost']).sum()
                        st.metric(
                            "Total Investment",
                            f"{currency_symbol}{total_investment:,.0f}"
                        )

                    with col4:
                        marginal_cost = needed_measures['cost'].max()
                        st.metric(
                            "Marginal Cost",
                            f"{currency_symbol}{marginal_cost:.2f}/{co2_unit}"
                        )

                    # Visualization
                    fig_target = go.Figure()

                    for i in range(len(df_calc)):
                        x_start = df_calc['cumulative_abatement'].iloc[i-1] if i > 0 else 0
                        x_end = df_calc['cumulative_abatement'].iloc[i]
                        y = df_calc['cost'].iloc[i]

                        # Check if this measure is needed
                        is_needed = df_calc.index[i] in needed_measures.index

                        if is_needed:
                            color = 'rgba(46, 125, 50, 0.7)'
                        else:
                            color = 'rgba(189, 189, 189, 0.3)'

                        fig_target.add_trace(go.Scatter(
                            x=[x_start, x_end, x_end, x_start, x_start],
                            y=[0, 0, y, y, 0],
                            fill='toself',
                            fillcolor=color,
                            line=dict(color=color.replace('0.7', '1.0').replace('0.3', '0.8'), width=1),
                            showlegend=False,
                            hovertemplate=f'Cost: {currency_symbol}{y:.2f}/{co2_unit}<br>' +
                                        f'Status: {"Required" if is_needed else "Not Required"}<extra></extra>'
                        ))

                    # Add target abatement line
                    fig_target.add_vline(
                        x=target_abatement,
                        line_dash="dash",
                        line_color="red",
                        line_width=3,
                        annotation_text=f"Target: {target_abatement:,.0f} {co2_unit}",
                        annotation_position="top"
                    )

                    fig_target.update_layout(
                        title=f"Required Measures for {target_abatement:,.0f} {co2_unit} Abatement",
                        xaxis_title=f"Cumulative Abatement Potential ({co2_unit})",
                        yaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                        height=500
                    )

                    st.plotly_chart(fig_target, use_container_width=True)

                    # List of required measures
                    if len(needed_measures) > 0:
                        st.subheader("Required Measures")
                        needed_display = needed_measures[['measure', 'cost', 'abatement']].copy()
                        needed_display['total_cost'] = needed_display['cost'] * needed_display['abatement']
                        needed_display.columns = ['Measure', f'Cost ({currency_symbol}/{co2_unit})', f'Abatement ({co2_unit})', f'Total Cost ({currency_symbol})']
                        st.dataframe(needed_display, use_container_width=True, hide_index=True)

                # Multiple scenario comparison
                st.markdown("---")
                st.subheader("Multi-Scenario Comparison")

                # Define scenarios
                scenarios = {
                    "Conservative": implicit_price * 0.5,
                    "Moderate": implicit_price,
                    "Aggressive": implicit_price * 1.5,
                    "Very Aggressive": implicit_price * 2.0
                }

                scenario_results = []

                for scenario_name, price in scenarios.items():
                    viable = df_calc[df_calc['cost'] <= price]
                    scenario_results.append({
                        'Scenario': scenario_name,
                        'Price Target': f"{currency_symbol}{price:.2f}",
                        'Measures': len(viable),
                        'Abatement': viable['abatement'].sum(),
                        'Investment': (viable['abatement'] * viable['cost']).sum()
                    })

                scenario_df = pd.DataFrame(scenario_results)
                total_abatement = df_calc['abatement'].sum()
                scenario_df['Abatement %'] = ((scenario_df['Abatement'] / total_abatement * 100) if total_abatement > 0 else 0).round(1)

                st.dataframe(scenario_df, use_container_width=True, hide_index=True)

                # Scenario comparison chart
                fig_compare = go.Figure()

                fig_compare.add_trace(go.Bar(
                    x=scenario_df['Scenario'],
                    y=scenario_df['Abatement'],
                    name='Abatement Potential',
                    marker_color='rgba(46, 125, 50, 0.7)'
                ))

                fig_compare.update_layout(
                    title="Abatement Potential by Scenario",
                    xaxis_title="Scenario",
                    yaxis_title=f"Abatement ({co2_unit})",
                    height=400
                )

                st.plotly_chart(fig_compare, use_container_width=True)

        # Tab 5: Statistical Analysis
        with tab5:
            st.header("Statistical Analysis")

            if st.session_state.df_calc is not None:
                df_calc = st.session_state.df_calc

                if len(df_calc) == 0:
                    st.error("No data available after filtering. Please adjust your filters or upload different data.")
                    st.stop()

                # Correlation analysis
                st.subheader("Correlation Analysis")

                corr = df_calc[['abatement', 'cost']].corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 16}
                ))

                fig_corr.update_layout(
                    title="Correlation Matrix",
                    height=400
                )

                st.plotly_chart(fig_corr, use_container_width=True)

                correlation_value = corr.loc['abatement', 'cost']

                if abs(correlation_value) < 0.3:
                    correlation_strength = "weak"
                elif abs(correlation_value) < 0.7:
                    correlation_strength = "moderate"
                else:
                    correlation_strength = "strong"

                st.info(f"The correlation between abatement potential and cost is **{correlation_strength}** (r = {correlation_value:.3f})")

                # Distribution analysis
                st.subheader("Distribution Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Abatement distribution
                    fig_abate_dist = go.Figure()

                    fig_abate_dist.add_trace(go.Histogram(
                        x=df_calc['abatement'],
                        nbinsx=20,
                        marker_color='rgba(46, 125, 50, 0.7)',
                        name='Abatement'
                    ))

                    fig_abate_dist.update_layout(
                        title="Abatement Distribution",
                        xaxis_title=f"Abatement ({co2_unit})",
                        yaxis_title="Frequency",
                        height=400
                    )

                    st.plotly_chart(fig_abate_dist, use_container_width=True)

                    # Abatement statistics
                    st.markdown("**Abatement Statistics**")
                    abate_stats = {
                        "Mean": df_calc['abatement'].mean(),
                        "Median": df_calc['abatement'].median(),
                        "Std Dev": df_calc['abatement'].std(),
                        "Skewness": df_calc['abatement'].skew(),
                        "Kurtosis": df_calc['abatement'].kurtosis()
                    }
                    st.json(abate_stats)

                with col2:
                    # Cost distribution
                    fig_cost_dist = go.Figure()

                    fig_cost_dist.add_trace(go.Histogram(
                        x=df_calc['cost'],
                        nbinsx=20,
                        marker_color='rgba(33, 150, 243, 0.7)',
                        name='Cost'
                    ))

                    fig_cost_dist.update_layout(
                        title="Cost Distribution",
                        xaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                        yaxis_title="Frequency",
                        height=400
                    )

                    st.plotly_chart(fig_cost_dist, use_container_width=True)

                    # Cost statistics
                    st.markdown("**Cost Statistics**")
                    cost_stats = {
                        "Mean": df_calc['cost'].mean(),
                        "Median": df_calc['cost'].median(),
                        "Std Dev": df_calc['cost'].std(),
                        "Skewness": df_calc['cost'].skew(),
                        "Kurtosis": df_calc['cost'].kurtosis()
                    }
                    st.json(cost_stats)

                # Scatter plot
                st.subheader("Cost vs. Abatement Relationship")

                fig_scatter = go.Figure()

                fig_scatter.add_trace(go.Scatter(
                    x=df_calc['abatement'],
                    y=df_calc['cost'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=df_calc['cost'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=f"Cost<br>({currency_symbol}/{co2_unit})")
                    ),
                    text=df_calc['measure'],
                    hovertemplate='<b>%{text}</b><br>' +
                                f'Abatement: %{{x:.0f}} {co2_unit}<br>' +
                                f'Cost: {currency_symbol}%{{y:.2f}}/{co2_unit}<extra></extra>'
                ))

                # Add trend line
                z = np.polyfit(df_calc['abatement'], df_calc['cost'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df_calc['abatement'].min(), df_calc['abatement'].max(), 100)

                fig_scatter.add_trace(go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                ))

                fig_scatter.update_layout(
                    title="Cost vs. Abatement Scatter Plot",
                    xaxis_title=f"Abatement Potential ({co2_unit})",
                    yaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                    height=500
                )

                st.plotly_chart(fig_scatter, use_container_width=True)

                # Percentile analysis
                st.subheader("Percentile Analysis")

                percentiles = [10, 25, 50, 75, 90, 95, 99]
                percentile_values = df_calc['cost'].quantile([p/100 for p in percentiles])

                percentile_df = pd.DataFrame({
                    'Percentile': [f"{p}th" for p in percentiles],
                    'Cost': [f"{currency_symbol}{v:.2f}/{co2_unit}" for v in percentile_values.values]
                })

                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_percentile = go.Figure()

                    fig_percentile.add_trace(go.Bar(
                        x=[f"{p}th" for p in percentiles],
                        y=percentile_values.values,
                        marker_color='rgba(46, 125, 50, 0.7)'
                    ))

                    fig_percentile.update_layout(
                        title="Cost Percentiles",
                        xaxis_title="Percentile",
                        yaxis_title=f"Cost ({currency_symbol}/{co2_unit})",
                        height=400
                    )

                    st.plotly_chart(fig_percentile, use_container_width=True)

                with col2:
                    st.dataframe(percentile_df, use_container_width=True, hide_index=True)

        # Tab 6: Export & Reports
        with tab6:
            st.header("Export & Reports")

            if st.session_state.df_calc is not None:
                df_calc = st.session_state.df_calc

                if len(df_calc) == 0:
                    st.error("No data available after filtering. Please adjust your filters or upload different data.")
                    st.stop()

                st.subheader("Report Summary")

                # Generate comprehensive report
                report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.markdown(f"""
                ### Carbon Pricing Analysis Report
                **Generated:** {report_date}

                **Dataset Summary:**
                - Total Measures: {len(df_calc)}
                - Total Abatement Potential: {df_calc['abatement'].sum():,.0f} {co2_unit}
                - Cost Range: {currency_symbol}{df_calc['cost'].min():.2f} to {currency_symbol}{df_calc['cost'].max():.2f} per {co2_unit}

                **Carbon Pricing Metrics:**
                - Internal Carbon Price (ICP): {currency_symbol}{icp:.2f}/{co2_unit}
                - Shadow Price of Carbon (SPC): {currency_symbol}{spc:.2f}/{co2_unit}
                - Implicit Carbon Price: {currency_symbol}{implicit_price:.2f}/{co2_unit}

                **Investment Summary:**
                - Total Investment Required: {currency_symbol}{total_cost_investment:,.0f}
                - Negative Cost Measures: {len(df_calc[df_calc['cost'] < 0])}
                - Net Savings from Negative Cost: {currency_symbol}{abs((df_calc[df_calc['cost'] < 0]['abatement'] * df_calc[df_calc['cost'] < 0]['cost']).sum()):,.0f}
                """)

                st.markdown("---")

                # Export options
                st.subheader("Export Options")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Export Pricing Metrics**")

                    metrics_df = pd.DataFrame({
                        'Metric': [
                            'Internal Carbon Price (ICP)',
                            'Shadow Price of Carbon (SPC)',
                            'Implicit Carbon Price',
                            'Mean Cost',
                            'Median Cost',
                            'Min Cost',
                            'Max Cost',
                            'Std Deviation'
                        ],
                        f'Value ({currency_symbol}/{co2_unit})': [
                            icp,
                            spc,
                            implicit_price,
                            df_calc['cost'].mean(),
                            df_calc['cost'].median(),
                            df_calc['cost'].min(),
                            df_calc['cost'].max(),
                            df_calc['cost'].std()
                        ]
                    })

                    metrics_csv = metrics_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Metrics (CSV)",
                        data=metrics_csv,
                        file_name=f"carbon_pricing_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col2:
                    st.markdown("**Export Processed Data**")

                    export_df = df_calc[['measure', 'cost', 'abatement', 'cumulative_abatement']].copy()
                    export_df['total_cost'] = export_df['cost'] * export_df['abatement']
                    export_df.columns = [
                        'Measure',
                        f'Cost ({currency_symbol}/{co2_unit})',
                        f'Abatement ({co2_unit})',
                        f'Cumulative Abatement ({co2_unit})',
                        f'Total Cost ({currency_symbol})'
                    ]

                    data_csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Full Data (CSV)",
                        data=data_csv,
                        file_name=f"macc_data_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                # Comprehensive report
                st.markdown("---")
                st.subheader("Comprehensive Report")

                comprehensive_report = f"""
CARBON PRICING ANALYSIS REPORT
===============================
Generated: {report_date}

1. DATASET OVERVIEW
-------------------
Total Measures: {len(df_calc)}
Total Abatement Potential: {df_calc['abatement'].sum():,.2f} {co2_unit}
Total Investment Required: {currency_symbol}{total_cost_investment:,.2f}

2. CARBON PRICING METRICS
--------------------------
Internal Carbon Price (ICP): {currency_symbol}{icp:.2f}/{co2_unit}
  Calculation Method: {calculation_method}
  Description: {icp_description}

Shadow Price of Carbon (SPC): {currency_symbol}{spc:.2f}/{co2_unit}
  Description: Marginal cost of the last abatement measure implemented

Implicit Carbon Price: {currency_symbol}{implicit_price:.2f}/{co2_unit}
  Description: Median cost across all abatement measures

3. COST STATISTICS
------------------
Mean Cost: {currency_symbol}{df_calc['cost'].mean():.2f}/{co2_unit}
Median Cost: {currency_symbol}{df_calc['cost'].median():.2f}/{co2_unit}
Std Deviation: {currency_symbol}{df_calc['cost'].std():.2f}/{co2_unit}
Min Cost: {currency_symbol}{df_calc['cost'].min():.2f}/{co2_unit}
Max Cost: {currency_symbol}{df_calc['cost'].max():.2f}/{co2_unit}

Quartiles:
  Q1 (25th percentile): {currency_symbol}{df_calc['cost'].quantile(0.25):.2f}/{co2_unit}
  Q2 (50th percentile): {currency_symbol}{df_calc['cost'].quantile(0.50):.2f}/{co2_unit}
  Q3 (75th percentile): {currency_symbol}{df_calc['cost'].quantile(0.75):.2f}/{co2_unit}

4. COST BREAKDOWN
-----------------
Negative Cost Measures: {len(df_calc[df_calc['cost'] < 0])}
  Total Abatement: {df_calc[df_calc['cost'] < 0]['abatement'].sum():,.2f} {co2_unit}
  Net Savings: {currency_symbol}{abs((df_calc[df_calc['cost'] < 0]['abatement'] * df_calc[df_calc['cost'] < 0]['cost']).sum()):,.2f}

Low Cost Measures (below median): {len(df_calc[(df_calc['cost'] >= 0) & (df_calc['cost'] < implicit_price)])}
  Total Abatement: {df_calc[(df_calc['cost'] >= 0) & (df_calc['cost'] < implicit_price)]['abatement'].sum():,.2f} {co2_unit}
  Total Cost: {currency_symbol}{(df_calc[(df_calc['cost'] >= 0) & (df_calc['cost'] < implicit_price)]['abatement'] * df_calc[(df_calc['cost'] >= 0) & (df_calc['cost'] < implicit_price)]['cost']).sum():,.2f}

High Cost Measures (above median): {len(df_calc[df_calc['cost'] >= implicit_price])}
  Total Abatement: {df_calc[df_calc['cost'] >= implicit_price]['abatement'].sum():,.2f} {co2_unit}
  Total Cost: {currency_symbol}{(df_calc[df_calc['cost'] >= implicit_price]['abatement'] * df_calc[df_calc['cost'] >= implicit_price]['cost']).sum():,.2f}

5. TOP MEASURES
---------------
Most Cost-Effective (Top 5):
{chr(10).join([f"  {i+1}. {row['measure']}: {currency_symbol}{row['cost']:.2f}/{co2_unit} ({row['abatement']:.0f} {co2_unit})" for i, row in df_calc.nsmallest(5, 'cost').iterrows()])}

Most Expensive (Top 5):
{chr(10).join([f"  {i+1}. {row['measure']}: {currency_symbol}{row['cost']:.2f}/{co2_unit} ({row['abatement']:.0f} {co2_unit})" for i, row in df_calc.nlargest(5, 'cost').iterrows()])}

6. CONFIGURATION
----------------
Currency: {currency}
CO2 Unit: {co2_unit}
Calculation Method: {calculation_method}
Include Negative Costs: {include_negative_costs}

===============================
End of Report
"""

                st.text_area("Report Preview", comprehensive_report, height=400)

                st.download_button(
                    label="📄 Download Full Report (TXT)",
                    data=comprehensive_report,
                    file_name=f"carbon_pricing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        with st.expander("Show Error Details"):
            st.code(traceback.format_exc())

else:
    # Landing page when no file is uploaded
    st.info("👈 Please upload a CSV or Excel file to begin analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("About This Tool")
        st.markdown("""
        This **Carbon Pricing Calculator** helps organizations analyze their carbon abatement opportunities
        and calculate key carbon pricing metrics including:

        - **Internal Carbon Price (ICP)**: Your organization's weighted average cost of carbon abatement
        - **Shadow Price of Carbon (SPC)**: The marginal cost of your last abatement measure
        - **Implicit Carbon Price**: The median cost across all measures

        ### Features:
        - 📊 Interactive MACC (Marginal Abatement Cost Curve) visualization
        - 🎯 Scenario analysis with carbon price and abatement targets
        - 📈 Comprehensive statistical analysis
        - 📥 Multiple export formats for reports and data
        - ⚙️ Flexible calculation methodologies
        - 🔍 Advanced filtering and data preprocessing

        ### How to Use:
        1. Upload your MACC data (CSV or Excel format)
        2. Map your data columns to the required fields
        3. Configure calculation settings in the sidebar
        4. Explore the interactive tabs for detailed analysis
        5. Export your results and reports
        """)

    with col2:
        st.subheader("Data Format Guide")
        st.markdown("""
        Your data should contain **at least two columns**:

        **Required:**
        - Abatement Potential (tCO2e)
        - Cost per Unit ($/tCO2e)

        **Optional:**
        - Measure/Project Name
        - Category/Sector
        - Year/Time Period
        """)

    # Sample data
    st.subheader("Sample Data Format")

    sample_data = pd.DataFrame({
        'Measure': [
            'LED Lighting Upgrade',
            'HVAC Optimization',
            'Solar Panel Installation',
            'Building Insulation',
            'Electric Vehicle Fleet',
            'Wind Energy Purchase',
            'Waste Heat Recovery',
            'Process Optimization',
            'Carbon Capture',
            'Renewable Energy Credits'
        ],
        'Abatement (tCO2e)': [500, 800, 1200, 600, 1500, 2000, 400, 700, 1000, 900],
        'Cost ($/tCO2e)': [15, 22, 35, 18, 45, 50, 12, 28, 75, 38],
        'Category': ['Energy', 'Energy', 'Renewable', 'Energy', 'Transport', 'Renewable', 'Energy', 'Process', 'Technology', 'Renewable']
    })

    st.dataframe(sample_data, use_container_width=True, hide_index=True)

    # Download sample data
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Sample Data",
        data=sample_csv,
        file_name="sample_macc_data.csv",
        mime="text/csv"
    )

    st.markdown("---")

    # Additional information
    with st.expander("ℹ️ About Carbon Pricing Metrics"):
        st.markdown("""
        ### Internal Carbon Price (ICP)
        An internal carbon price is a monetary value assigned to carbon emissions by an organization to guide
        investment decisions and drive emissions reductions. It can be calculated using different methods:

        - **Weighted Average**: Total cost divided by total abatement (considers size of each measure)
        - **Simple Average**: Mean cost across all measures (treats all measures equally)
        - **Marginal Cost**: Cost of the last measure implemented (represents the price needed for full abatement)

        ### Shadow Price of Carbon (SPC)
        The shadow price represents the economic cost of reducing one additional unit of carbon emissions.
        It's typically the marginal cost of the most expensive abatement measure implemented.

        ### Implicit Carbon Price
        This represents a middle-ground price point (median) that balances low-cost and high-cost measures.
        It's useful for understanding the typical cost of abatement in your portfolio.

        ### MACC (Marginal Abatement Cost Curve)
        A MACC is a visual tool that shows the cost and potential of various carbon abatement measures.
        Measures are ordered from lowest to highest cost, creating a curve that helps prioritize investments.
        """)

    with st.expander("📚 Best Practices"):
        st.markdown("""
        ### Data Quality
        - Ensure cost data is in consistent units ($/tCO2e or similar)
        - Remove or flag outliers that may skew results
        - Verify abatement potential calculations are accurate
        - Consider including uncertainty ranges for costs and abatement

        ### Methodology Selection
        - **Weighted Average**: Best for overall portfolio assessment
        - **Simple Average**: Useful when all measures have similar scale
        - **Marginal Cost**: Important for understanding incremental decision-making

        ### Scenario Analysis
        - Test multiple carbon price scenarios to understand flexibility
        - Analyze abatement targets aligned with your climate goals
        - Consider time horizons (some measures may have different costs over time)

        ### Reporting
        - Document your methodology and assumptions
        - Compare your ICP to market carbon prices and regulatory requirements
        - Update your MACC regularly as costs and technologies evolve
        """)
