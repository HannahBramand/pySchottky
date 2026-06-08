import numpy as np

def momentum_with_RF_data(f_rev, num_turns, slip_fac_eta, dp_p_list, V_rf=0.002):
    """
    V_rf: قدرت ولتاژ کاواک RF (نیروی بازدارنده)
    """
    T_rev = 1.0 / f_rev
    
    all_pass_times = []
    all_dp_p_history = []
    
    # برای هر ذره با مومنتوم اولیه متفاوت، یک شبیه‌سازی جداگانه انجام می‌دهیم
    for initial_dp in dp_p_list:
        
        current_dt = 0.0          # اختلاف زمانی اولیه (صفر)
        current_dp = initial_dp   # خطای مومنتوم اولیه
        current_abs_time = 0.0    # زمان واقعی عبور
        
        pass_times = []
        dp_history = []
        
        # شبیه‌سازی دور به دور (Turn-by-turn tracking)
        for n in range(num_turns):
            # الف) اثر RF: تغییر مومنتوم بر اساس فاز (زمان رسیدن)
            # اگر ذره دیر یا زود برسد، انرژی متفاوتی از کاواک می‌گیرد
            phase = 2 * np.pi * f_rev * current_dt
            current_dp = current_dp - V_rf * np.sin(phase)
            
            # ب) اثر لغزش (Drift): تغییر زمان رسیدن بر اساس مومنتوم جدید
            current_dt = current_dt + T_rev * slip_fac_eta * current_dp
            
            # ج) محاسبه زمان واقعی رسیدن به سنسور برای رسم نمودار بالا
            current_abs_time = current_abs_time + T_rev * (1 + slip_fac_eta * current_dp)
            
            pass_times.append(current_abs_time)
            dp_history.append(current_dp) # ذخیره مومنتوم که حالا در حال تغییر است
            
        all_pass_times.append(np.array(pass_times))
        all_dp_p_history.append(np.array(dp_history))
        
    # حالا تابع دو چیز برمی‌گرداند: زمان‌های عبور و تاریخچه تغییرات مومنتوم
    return all_pass_times, all_dp_p_history