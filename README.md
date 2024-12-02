# The Future of the Kingdom of Marendor 🔮

### Team Members
- Lina Kolevska
- Chloe Quevedo
- Elina Yilmaz

## The Question ⁉️:
Achievements, Qualities, Current Status: all features that determine the fate of these competing scholars. This brings us to the final, dire question:
### **Who will be worthy enough to ascend into the Master Guild of Magic?**


## Our Task 💭:
With competition at an all time high within the kingdom, a race to contribute the most to the technological development and progression of Marendor is in commencement. 

* Their paramount goal is to utilize their overall potential to reach the final, and most determinant level of wizardry: the `"Master Guild”.`

* Most scholars begin in the lowly phases of magic, with the heavy majority of scholars **not** belonging to any Guild; their Guild status is simple `“No Guild”.`

* A much smaller fraction of the remaining scholars, whom of which have not ascended into professional magic, have managed to get into the second-tier guild: the `“Apprentice Guild”.` Here, the scholars are in a preparatory phase to reach a Master Guild status. 

### Considering these three membership situations: we determined that the task at hand requires a: `multi-class classification approach.`



Commissioned by the King of Marendor, we were granted publicly-shared, personal details about these determined scholars. With this data, we trained the most efficient, top-of-the-line models following a thorough process of cleaning, balancing, and incrementally analyzing the supplied data. 

By collectively taking every feature, whether physiological attributes or magic-capability measurement, we determine the most effective methodologies/approaches to maximize predicting potential! Given the multitude of features to analyze from, we harnessed these strategies to hunt for the most relevant and useful features to accomplish this classification task.

## Project Preparation 🧮:
*In efforts to manipulate and assess the dataset to its greatest capacity, we employed a myriad of Python libraries, a majority of functions coming from Scikit-Learn, and many others from Pandas, NumPy, Seaborn, Missingno, PyPlot, SciPy, and we utilized Python modules like “time” and “collections”.


**Purpose of Libraries & Modules 📚:**
- `Scikit-Learn:` offers tools for classification, regression, clustering, and dimensionality reduction. This library contains algorithms like SVMs, Random Forests, K-Means Clustering and has tools for evaluating our models and preprocessing the data.
- `Pandas:` integral for data manipulation and analysis and grants data structure like “DataFrame”, especially for tabular data. This helps clean data, aggregating/merging, and filtering and handling missing values.
- `NumPy:` used for numerical computing and working to support large arrays with multidimensions and matrices. This completes statistical computations and linear algebra, while manipulating data on a quantitative scale.
- `Seaborn:` to statistically visualize data (working with Matplotlib) to effectively develop and output aesthetic plots. We implemented this greatly for our correlation heatmaps, our box plots, and more for our EDA
- `Missingno:` mainly employed to visualize our missing values and comprehend the patterns of our missing data. This is imperative in cleaning and preparing our dataset.
- `PyPlot:` as a module of the Matplotlib library, we utilized this to create interactive visualizations to plot out graphs and add labels, all while customizing our conceptualizations. 
- `time module:` This module is built into Python but worth mentioning as it deals with time-related problems and operations. We employed this module to measure execution time and establish timestamps.
- `collections module:` This is also a built-in Python module and offers specialized manipulations of collections of data. It is able to enumerate frequencies with its Counter function and create ordered dictionaries.


## A Step by Step Guide into the Project🔎:
### *Completing the Royal Task 🌟*
No monarchical projects go without meticulous preparation and pre-structuring, and thus, we configured a series of steps to complete our task, here they are in-depth:

### 1) Comprehending the structure of the Scholar data 🧱:
* 1.1) In order to understand what features and relationships we have to deal with, we commenced by describing and exploring our superficial data, with no edits or manipulation. We coin the **original_data** variable as our unedited, raw **guilds.csv**. We maintain the integrity of the original dataset to distinguish all future changes from the original set of information, protecting the initial data for comparison. 
* 1.2) It is important to note that the majority of our categorical values are in the pattern of *“Present”* versus *“Absent”*, clarifying the presence of certain attributes and their belonging to a designated scholar. Our **target variable** is the `Guild_Membership` feature, which ultimately determines the fate of these scholars and their predictions for guild classification. Within this target column, the values are given as *“No_Guild”*, *”Master_Guild”*, and *“Apprentice_Guild”*, as mentioned previously.
* 1.3) We begin by printing the Rows and Columns totals as our original data shape, display our data types, and print our missing value distributions. This is accomplished using a `.info( )` built-in function. Here we deem that our dataset is quite large, with **253,68**0 rows and **31 features** (columns). 
* 1.4) Then, by delineating our two types of data, numerical vs. categorical, we individually describe the feature type. This is accomplished by using the `.describe( )` built-in, one time without a parameter, and another run, including *‘object’* types, for the categorical data. 
* 1.5) We proceed with summing the presence of missing values, per column, and print this for display. Each feature has at least **25,000** missing values, causing an extreme gap in thoroughness, prompting us to note this and meticulously fix, throughout our EDA process. To ensure that no feature is *too* absent for analysis, we also performed a missing value rate for each column; we display this rate in a descending order. The missing value rate does not exceed **10.12%**, and thus we decided to maintain every feature. We determined the main goal was to fill these missing values properly, an intensive task to achieve with such a large dataset.
* 1.6) Now, to address the scholars and their documentation, we calculated a missing rate, again, but this time for each *row*. Working again off the original dataset, we find the largest and smallest number of missing values per row, and store these in separate variables, to be printed. Our maximum missing value count is **14**, and our minimum is **0**. The majority of rows contain **1 to 5 missing values**, making the dataset relatively complete. This is a decent range and is extremely indicative for future data reduction and consolidation. We proceed to group our Scholars by missing value total and plot these with a `.plot(kind=’bar’, color=’purple’, alpha=0.1)` function with our x-axis being the amounts of missing values, and the y-axis being the amount of rows. This distribution of Missing Values allows us to conceptualize spatially.  
<img width="648" alt="Screenshot 2024-12-02 at 14 39 47" src="https://github.com/user-attachments/assets/7f11b0a3-1e64-43d5-82d3-844a4f4fb6d5">



### 2) Performing an extensive Exploratory Data Analysis: Broken Down in Parts
* 2.1) At this point, we have retrieved the basic information and distributions of our original dataset and decide to proceed with the cleaning and manipulation. We determined it necessary to copy the *original_data* into a dataframe; we call it **”df”**. This is accomplished with the *.copy( )* built-in. Again, this is done for future comparison and analysis amongst intervals of data modifications. 
* 2.2) Previously identifying the target variable `Guild_Membership`, it is paramount that the status of this feature is not null for each examined Scholar. Thus, we remove the Scholars that have a missing membership status, allowing us to properly analyze and handle our task. To accomplish this we print the missing value count for each Scholar (each row), within the Guild Membership column. In the case that the value is missing, we drop it with a `.dropna(subset=[‘Guild_Membership’])`. To verify this removal, we print the summed Guild_membership missing values, post-removal. We began with **25,447** Scholars with zero Membership documentation, and prevailed with **0** undocumented Scholars. 
* 2.3) Focusing in on our target variable, we proceeded with delving into its distribution. Utilizing a `.value_counts( )` built-in, we aggregated the total number of presences, per class. Our results:
  * `No_Guild`: 192,328
  * `Master_Guild`: 31,739
  * `Apprentice_Guild`: 4,166.
* It is clear that the majority of our Scholars (75.8% to be precise) have not yet managed to enter a Guild, consequently imbalancing our data greatly. We visualize this distribution with a `.countplot(data=df, x=’Guild_Membership’)` built-in
<img width="599" alt="Screenshot 2024-12-02 at 15 15 48" src="https://github.com/user-attachments/assets/b7253b96-1547-40c6-b5ec-2ffa0b8a8047">


* 2.4) Working over two types of Scholar Feature, categorical and numerical, we separate the two for individual analysis. We create two respective variables for these column types and use a `.select_types(include=[‘object’, ‘bool’])` and a `.select_types(include=[‘float64’, ‘int64’])` built-in. We initialize lists for the discrete and continuous numerical values, further splitting one of the data types into categories. By establishing a for loop for each numerical column, we count unique values with a `.nunique( )` built-in. We assume that if the value type within each column is an integer or has less than 20 unique values, they can be categorized as *discrete* and added to the respective list, while all other cases are continuous values and are consequently added to the continuous list. These 4 data types: categorical, numerical, discrete, and continuous data are separated into dataframes, obviously numerical data having an overlap amongst the numerical category dataframes (continuous and discrete). By doing this we list the data type categories and display the characteristics of each feature. We retrieve the following:
  * `Categorical Columns:` 'Healer_consultation_Presence', 'Elixir_veggies_consumption_Presence', 'Bolt_of_doom_Presence', 'High_willingness_Presence', 'Defense_spell_difficulty_Presence', 'Doc_availability_challenge_Presence', 'Dexterity_check_Presence', 'Fruits_of_eden_consumption_Presence', 'Knight_physical_training_Presence', 'Royal_family_pressure_Presence', 'Guild_Membership', 'Heavy_elixir_consumption_Presence', 'Stigmata_of_the_cursed_Presence', 'Dragon_status_Presence' 
  * `Numerical Columns:` 'Fae_Dust_Reserve', 'Physical_Stamina', 'Mystical_Index', 'Mystic_Energy_Level', 'Age_of_Wisdom', 'Mental_Wizardry', 'Potion_Power_Level', 'Gold_Pouches_Per_Year', 'Wizardry_Skill', 'Spell_Mastering_Days', 'Level_of_Academic_Wisdom', 'General_Health_Condition', 'Dragon_Sight_Sharpness', 'Enchanted_Coin_Count', 'Celestial_Alignment', 'Knightly_Valor', 'Rune_Power', 'Age_of_Wisdom', 'Gold_Pouches_Per_Year', 'Level_of_Academic_Wisdom', 'General_Health_Condition', 'Dragon_Sight_Sharpness', 'Knightly_Valor'
  * `Discrete Columns:` 'Age_of_Wisdom', 'Gold_Pouches_Per_Year', 'Level_of_Academic_Wisdom', 'General_Health_Condition', 'Dragon_Sight_Sharpness', 'Knightly_Valor'
  * `Continuous Columns:` ‘Fae_Dust_Reserve', 'Physical_Stamina', 'Mystical_Index', 'Mystic_Energy_Level', 'Mental_Wizardry', 'Potion_Power_Level', 'Wizardry_Skill', 'Spell_Mastering_Days', 'Enchanted_Coin_Count', 'Celestial_Alignment', 'Rune_Power'
* 2.2)
* 2.3)
* 2.4) 
### 3) Model Preparation

### 4) Training and Testing Models 

### 5) Plotting our Learning Curves

### 6) Evaluating our Models

### 7) Hyperparameter Tuning

### 8) Hyperparameter Sensitivity Analysis:

### 9) Our Expert Cut: How Can We Optimize Guild Prediction?


