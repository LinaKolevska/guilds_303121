
## **<h2 align="center"> ★ The Future of the Kingdom of Marendor 🔮 ★ </h2>** 

### Team Members
- Lina Kolevska
- Chloe Quevedo
- Elina Yilmaz

## The Royal Question 👑:
Achievements, Qualities, Current Status: all features that determine the fate of these competing scholars. This brings us to the final, dire question:
### **Who will be worthy enough to ascend into the Master Guild of Magic?**


## Our Task 💭:
With competition at an *all time high* within the kingdom, a race to contribute the most to the technological development and progression of Marendor is in commencement. 

* The Scholars' paramount goal is to utilize their overall potential to reach the highest, and most determinant level of wizardry: the `"Master Guild”.`

* Most scholars begin in the lowly phases of magic, with the heavy majority of scholars **not** belonging to any Guild; their Guild status is simple `“No Guild”.`

* A much smaller fraction of the remaining scholars, whom of which have not ascended into professional magic, have managed to get into the second-tier guild: the `“Apprentice Guild”.` Here, the scholars are in a preparatory phase to reach a Master Guild status. 

### Considering these three membership situations: we determined that the task at hand requires a: `multi-class classification approach.`



Commissioned by the King of Marendor, we were granted publicly-shared, personal details about these determined scholars. With this data, we trained the most efficient, top-of-the-line models following a thorough process of cleaning, balancing, and incrementally analyzing the supplied data. 

By collectively taking every feature, whether physiological attributes or magic-capability measurement, we determine the most effective methodologies/approaches to maximize predicting potential! Given the multitude of features to analyze from, we harnessed these strategies to hunt for the most relevant and useful features to accomplish this classification task.

## Project Preparation 🧮:
*Under a Royal Contract, and in efforts to manipulate and assess the dataset to its greatest capacity, we employed a myriad of Python libraries, a majority of functions coming from Scikit-Learn, and many others from Pandas, NumPy, Seaborn, Missingno, PyPlot, SciPy, and we utilized Python modules like “time” and “collections”.


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
No **monarchical projects** go without meticulous preparation and pre-structuring, and thus, we configured a series of steps to complete our task, here they are in-depth:

## **<h4 align="center"> ★ 1) Comprehending the structure of the Scholar data 🧱: ★ </h4>** 
* 1.1) In order to understand what features and relationships we have to deal with, we commenced by describing and exploring our superficial data, with no edits or manipulation. We coin the **original_data** variable as our unedited, raw **guilds.csv**. We maintain the integrity of the original dataset to distinguish all future changes from the original set of information, protecting the initial data for comparison. 
* 1.2) It is important to note that the majority of our categorical values are in the pattern of *“Present”* versus *“Absent”*, clarifying the presence of certain attributes and their belonging to a designated scholar. Our **target variable** is the `Guild_Membership` feature, which ultimately determines the fate of these scholars and their predictions for guild classification. Within this target column, the values are given as *“No_Guild”*, *”Master_Guild”*, and *“Apprentice_Guild”*, as mentioned previously.
* 1.3) We begin by printing the Rows and Columns totals as our original data shape, display our data types, and print our missing value distributions. This is accomplished using a `.info( )` built-in function. Here we deem that our dataset is quite large, with **253,68**0 rows and **31 features** (columns). 
* 1.4) Then, by delineating our two types of data, numerical vs. categorical, we individually describe the feature type. This is accomplished by using the `.describe( )` built-in, one time without a parameter, and another run, including *‘object’* types, for the categorical data. 
* 1.5) We proceed with summing the presence of missing values, per column, and print this for display. Each feature has at least **25,000** missing values, causing an extreme gap in thoroughness, prompting us to note this and meticulously fix, throughout our EDA process. To ensure that no feature is *too* absent for analysis, we also performed a missing value rate for each column; we display this rate in a descending order. The missing value rate does not exceed **10.12%**, and thus we decided to maintain every feature. We determined the main goal was to fill these missing values properly, an intensive task to achieve with such a large dataset.
* 1.6) Now, to address the scholars and their documentation, we calculated a missing rate, again, but this time for each *row*. Working again off the original dataset, we find the largest and smallest number of missing values per row, and store these in separate variables, to be printed. Our maximum missing value count is **14**, and our minimum is **0**. The majority of rows contain **1 to 5 missing values**, making the dataset relatively complete. This is a decent range and is extremely indicative for future data reduction and consolidation. We proceed to group our Scholars by missing value total and plot these with a `.plot(kind=’bar’, color=’purple’, alpha=0.1)` function with our x-axis being the amounts of missing values, and the y-axis being the amount of rows. This distribution of Missing Values allows us to conceptualize spatially.  
<img width="648" alt="Screenshot 2024-12-02 at 14 39 47" src="https://github.com/user-attachments/assets/7f11b0a3-1e64-43d5-82d3-844a4f4fb6d5">



## **<h4 align="center"> ★ 2) Performing an extensive Exploratory Data Analysis: Broken Down in Parts ★ </h4>** 
#### Part I of Step 2: Cleaning
* 2.1.1) At this point, we have retrieved the basic information and distributions of our original dataset and decide to proceed with the cleaning and manipulation. We determined it necessary to copy the *original_data* into a dataframe; we call it **”df”**. This is accomplished with the *.copy( )* built-in. Again, this is done for future comparison and analysis amongst intervals of data modifications. 
* 2.1.2) Previously identifying the target variable `Guild_Membership`, it is paramount that the status of this feature is not null for each examined Scholar. Thus, we remove the Scholars that have a missing membership status, allowing us to properly analyze and handle our task. To accomplish this we print the missing value count for each Scholar (each row), within the Guild Membership column. In the case that the value is missing, we drop it with a `.dropna(subset=[‘Guild_Membership’])`. To verify this removal, we print the summed Guild_membership missing values, post-removal. We began with **25,447** Scholars with zero Membership documentation, and prevailed with **0** undocumented Scholars. 
* 2.1.3) Focusing in on our target variable, we proceeded with delving into its distribution. Utilizing a `.value_counts( )` built-in, we aggregated the total number of presences, per class. Our results:
  * `No_Guild`: 192,328
  * `Master_Guild`: 31,739
  * `Apprentice_Guild`: 4,166.
* It is clear that the majority of our Scholars (75.8% to be precise) have not yet managed to enter a Guild, consequently imbalancing our data greatly. We visualize this distribution with a `.countplot(data=df, x=’Guild_Membership’)` built-in
<img width="599" alt="Screenshot 2024-12-02 at 15 15 48" src="https://github.com/user-attachments/assets/b7253b96-1547-40c6-b5ec-2ffa0b8a8047">

* 2.1.4) Working over two types of Scholar Feature, categorical and numerical, we separate the two for individual analysis. We create two respective variables for these column types and use a `.select_types(include=[‘object’, ‘bool’])` and a `.select_types(include=[‘float64’, ‘int64’])` built-in. We initialize lists for the discrete and continuous numerical values, further splitting one of the data types into categories. By establishing a for loop for each numerical column, we count unique values with a `.nunique( )` built-in. We assume that if the value type within each column is an integer or has less than 20 unique values, they can be categorized as *discrete* and added to the respective list, while all other cases are continuous values and are consequently added to the continuous list. These 4 data types: categorical, numerical, discrete, and continuous data are separated into dataframes, obviously numerical data having an overlap amongst the numerical category dataframes (continuous and discrete). By doing this we list the data type categories and display the characteristics of each feature. We retrieve the following:
  * `Categorical Columns:` 'Healer_consultation_Presence', 'Elixir_veggies_consumption_Presence', 'Bolt_of_doom_Presence', 'High_willingness_Presence', 'Defense_spell_difficulty_Presence', 'Doc_availability_challenge_Presence', 'Dexterity_check_Presence', 'Fruits_of_eden_consumption_Presence', 'Knight_physical_training_Presence', 'Royal_family_pressure_Presence', 'Guild_Membership', 'Heavy_elixir_consumption_Presence', 'Stigmata_of_the_cursed_Presence', 'Dragon_status_Presence' 
  * `Numerical Columns:` 'Fae_Dust_Reserve', 'Physical_Stamina', 'Mystical_Index', 'Mystic_Energy_Level', 'Age_of_Wisdom', 'Mental_Wizardry', 'Potion_Power_Level', 'Gold_Pouches_Per_Year', 'Wizardry_Skill', 'Spell_Mastering_Days', 'Level_of_Academic_Wisdom', 'General_Health_Condition', 'Dragon_Sight_Sharpness', 'Enchanted_Coin_Count', 'Celestial_Alignment', 'Knightly_Valor', 'Rune_Power', 'Age_of_Wisdom', 'Gold_Pouches_Per_Year', 'Level_of_Academic_Wisdom', 'General_Health_Condition', 'Dragon_Sight_Sharpness', 'Knightly_Valor'
  * `Discrete Columns:` 'Age_of_Wisdom', 'Gold_Pouches_Per_Year', 'Level_of_Academic_Wisdom', 'General_Health_Condition', 'Dragon_Sight_Sharpness', 'Knightly_Valor'
  * `Continuous Columns:` ‘Fae_Dust_Reserve', 'Physical_Stamina', 'Mystical_Index', 'Mystic_Energy_Level', 'Mental_Wizardry', 'Potion_Power_Level', 'Wizardry_Skill', 'Spell_Mastering_Days', 'Enchanted_Coin_Count', 'Celestial_Alignment', 'Rune_Power'


#### Part II of Step 2: Descriptive Statistics
* 2.2.1) We now proceed with examining the Numerical Columns, by visualizing the distributions of the columns to uncover patterns, outliers, and data properties.
	* 1. **Dynamic Layout:** Rows and columns are calculated based on the number of variables to optimize space. This is done for future plotting. We utilize `ceil` logic as a security measure, to ensure that the columns fit within the grid. We create subplots with `plt.subplots(n_rows, n_cols)` to create a grid of subplots. These subplots individually display the numerical column distributions (for each feature). 
	* 2. **Histograms with KDE:** Utilizing `sns.histplot , we have a histogram with a kernel density estimate (KDE) overlay. We set `bins` to 30 and add a KDE curve. Histograms reveal frequency, while KDE overlays smooth density estimation for better interpretation.
	* 3. **Customization:** Titles, labels, and consistent formatting improve readability.
	* 4. **Clean Layout:** Extra empty plots are removed, and we employ `plt.tight_layout()` to adjust the subplot spacing. `plt.show` is used to plot everything.
	* **INSIGHT**: We notice that we have resulting skewed distributions, for example, our *Fae Dust Reserve* displays a right skew behavior. Several features like *Knightly Valor* have a distribution that suggests there could be the presence of subgroups. Features like *Potion Power Level* appear to have a Uniform distribution, while *Wizardry Skill* is almost Normal. We also noticed some black sheep columns, for example, the *Gold Pouches per Year* feature, with spikes in the distribution, possibly leading to a unique trend. Let it be known and understood that the *Power Potion Level* and *Enchanted Coin Count* have mostly flat/consistent distributions, which makes them high contenders for dropping, as they seem to be irrelevant to the future prediction 
* 2.2.2)  We now proceed by doing the same examinations, but for the Categorical Columns, by visualizing the distributions of the columns to uncover patterns, outliers, and data properties. We dynamically calculated the row numbers needed to the columns and again use the `ceil` logic to accomplish this. By creating subplots we are able to conceptualize the distribution subplots using a `plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 6))` built-in. We ensure to flatten the 2D axes array into one dimension for facilitated iteration. We display the category counts in each column with `value_counts( )` and output a bar chart to spatially visualize everything. We use `plt.tight_layout( )` again do avoid overlap.
	* **INSIGHT**: * We are able to see the comparisons of value presences, amongst the individual features. For example, columns similar to *Healear Consultation Presence* has far more *”Present”* values than *”Absent”* values (**195,303 vs. 10,106**). *Heavy Elixir Consumption Presence* and *Bolt of Doom*, “absences” absolutely dominate the features (**196,951 Absent  vs. 8,398 Present**). 
* The majority of our binary columns have severe class imbalances and the Multi-Category Influence (within the target variable), being so imbalanced, allows us to proceed into a specific type of encoding of the categorical variables. This led us to choose a mapping technique.

#### Part III of Step 2: Encoding Target Variable for Correlation Analyses 
* 2.3.1) Post analysis of both data types, proceeded with a mapping style encoding, customizing the values for each one of our guilds. This is obviously crucial to determine correlations and continue optimizing prediction performance. Instead of applying a one-hot or label encoding technique, we decide to customize our encoded values. We create a dictionary `guild_mapping` and allocate the following categorical values into encoded numbers:
	* `No_Guild` = 0
	* `Master_Guild` = 1
	* `Apprentice_Guild` = 2
* We proceed to apply this modification to the entire Guild Membership column, and map each value using a `.map( )` built-in, and proceed to store all of this into a brand new column called `Guild_Membership_encoded`. We later delete the original Guild_Membership column, and have it officially replaced by the encoded version. We proceed to segment our data based on these values: 0, 1, 2, allowing us to process each of these Guild classes separately. This way we can prepare for correlations and understand how each Guild class is impacted by the other features (essentially our target task). We utilize filtering conditions and store each of data subsets into three variables, one for each class.

* 2.3.2) Now, we get to mobilize the “segmenting” we just accomplished previously and apply it to an integral step of our EDA: *correlation*. We select all numerical columns, excluding the encoded Guild Membership column, using a `.difference( )` built-in, as the classes can not be correlated to themselves. We utilize the segmented data from above and compute the correlation matrices between each Guild Class and the numerical features and their varying values. This is accomplished with a `.corr( )` built-in. Outputted is a correlation matrix for each Guild class.

* **INSIGHT**: We observed the following from the matrices:
	- Rune_Power: We noticed that this feature consistently shows correlations close to 0 across all guild memberships, indicating it doesn’t provide meaningful relationships. 
	- Dragon_Sight_Sharpness: We observed that it has minimal correlation with other numerical features or the target variable, making it largely irrelevant.
	-  Spell_Mastering_Days: This feature has consistently low correlations across all guild categories, so we consider it uninformative.
	- Celestial_Alignment: Its relationships with Guild_Membership and other features are very weak, offering little value.
	-  Wizardry_Skill: We found that this feature has consistently weak correlations across all guilds and doesn’t contribute any significant or differentiating information.

* These analyses will be handled later, the important part is that we’ve identified many telling insights for our following actions.

* 2.3.3) In efforts to find the most significant correlation values quickly and efficiently, we extract the Top Correlated Feature per Guild class. We use `.abs( )` to extract the absolute values of the correlations, regardless of direction and convert the square matrix into a flat series of key-value pairs for each pair of features using `.unstack( )`; the key is the pair, the value if the absolute value. We ensure that no features are compared with themselves. These are sorted in descending order to get the top correlations and output the Top 10 Correlations. This is applied for each Guild. We gather the following observations:
	- `No_Guild`: Most Significant Correlations
		- `General_Health_Condition` ↔ `Physical_Stamina`: 0.4975
		- `Gold_Pouches_Per_Year` ↔ `Level_of_Academic_Wisdom`: 0.4351
		- General_Health_Condition appears a multitude of times in comparison with the Top Correlations, indicating that it is strongly correlated to other features like `Mental_Wizardry` and more, for the No Guild values. Because most of these correlations are involving health/academic metrics, we can gather that these values are essential in the No Guild causations
	- `Apprentice_Guild`: Most Significant Correlations
		- `Physical_Stamina` ↔ `General_Health_Condition`: 0.5310
		- `Gold_Pouches_Per_Year` ↔ `Level_of_Academic_Wisdom`: 0.4471
Health related features are also quite dominant in these correlations, however, some unique pairs of correlations appear here, for example `Age_of_Wisdom` ↔ `Mystical_Index` which is indicative for this specific class: potentially telling us that these Scholars are possibly older and have thus a concurrent Mystical Index that relates to their age status. 
	- `Master_Guild`: Most Significant Correlations
		- `General_Health_Condition` ↔ `Physical_Stamina`: 0.5446
		- `Level_of_Academic_Wisdom` ↔ `Gold_Pouches_Per_Year`: 0.4426
Similar to the Apprentice Guild, Age of Wisdom correlations seem important, in consistent measures, in associating potential importance in age with overall mystical experience. This guild class places slight emphasis on financial and mystical capacities, compared to the rest. All of these characteristics reflect the association of overall “maturity” with attaining a **Master Guild** status.

* 2.3.4) We decide to perform a Chi-Square Test of Independence, in an effort to identify the categorical features (excluding the target variable) that are associated with the target variable statistically. We employ statistical metrics and calculate the p-values for each feature. This also allows us to identify irrelevant features, taking a step closer to reduction and consolidation. We remove missing value present rows and create a contingency table to display the frequency of occurrences of each combination, using `pd.crosstab( )`. We follow this with a Chi-Square stat and p-value calculation using `chi2_contingency`. If the p-value is smaller, this implies a strong association between the target variables and features, and because of this, we sort by ascending value, prioritizing the smaller values. 
* **INSIGHT**: We observed the following from our outputs:
	**Highly Relevant Features**: `Bolt_of_doom_Presence`, `High_willingness_Presence`, and  `Defense_spell_difficulty_Presence` have extremely small values, with p-values of **0.00000e+00**
	**Moderately Significant Features**: `Fruits_of_eden_consumption_Presence`,`Doc_availability_challenge_Presence`, and `Healer_consultation_Presence` with still extremely small p-values, like **1.309751e-70**
The Highly Relevant Features are likely the most essential for Guild prediction, with heavy influence, and should be kept in mind for proceeding analyses.
	**Most Irrelevant Features**: We determined that the last three features, with the 3 largest p-values, are the most irrelevant amongst the categorical features. These are:
`Fruits_of_eden_consumption_Presence`, `Doc_availability_challenge_Presence`, `Healer_consultation_Presence`

#### Part IV of Step 2: Reducing our Dataset –based on our analysis of **Importance/Relevance**
* 2.4.1) Similarly done before, we copy our dataframe into a new one, called `df_reduced`. This way we can maintain the integrity of each change per important step, and have mediums to perform comparisons later on. Now we can perform continued data reduction and refinement. As identified just previously, we selected the 3 largest p-value features as the most irrelevant from the Top 10 most relevant selection; the rest of the features, non-included in the Top 10 results, will obviously be removed, as well, as their irrelevance is implied. This process drops 13 total irrelevant columns and prints the remaining ones.
* 2.4.2) We can finally continue reducing our dataset, commencing with the removal of the most irrelevant columns, then with the most undocumented Scholars, and finally with the rows that have missing values within the most “important/relevant” features. We identify the undocumented Scholars by selecting the ones that have missing values in more than 6 columns, establishing this as our threshold for removal –it is important to note that **we avoid all rows where Scholars are within the  `Apprentice_Guild` (value *2* within the encoded Guild Membership column), to protect the underrepresented Guild and its imbalance.** We print this shape, resulting in **227,700 Scholars** and **21 Features**. We have significantly reduced our dataset, but it is still far too large to *FILL* our missing values: our biggest objective to achieve before training our models. 

* 2.4.3) We proceed with determining the most relevant features. This time we drop the rows where there is *at least* one missing value among these features, in an effort to balance the dataset, while protecting the limited `Apprentice_Guild` values. We target the classes `No_Guild` and `Master_Guild` and their relevant features, because they are most prevalent in the database; they are heavily imbalanced in comparison with the `Apprentice_Guild`. The objective is to remove the rows with missing values to create a balanced dataset and ensure data quality within the test-training dataset. The relevant features identified are shown in the code as follows: **Physical_Stamina, General_Health_Condition, Gold_Pouches_Per_Year, Level_Of_Academic_Wisdom, Mental_Wizadry, 'Bolt_of_doom_Presence', 'High_willingness_Presence', Defense_spell_difficulty_Presence', Knight_physical_training_Presence, Royal_family_pressure_Presence, Stigmata_of_the_cursed_Presence**
There are two conditions encoded for identifying which rows to drop. These conditions are combined with `&` condition: 
`Guild_Membership_encoded == 0`: this filters the rows where guild membership is not present (*0*)
`.isnull().any(axis=1)`: this filters the rows where there are missing values (*NaN*)
	The DataFrame’s size after cleaning is reduced from (227,700, 21) to (95,828, 21) meaning **31,872**
rows were dropped. 
* 2.4.4) After looking at the distribution of the guild membership we see that it is still not balanced after doing all of the data cleaning. This allows us to quickly see the proportion of guild members versus non-members in the cleaned dataset. We also aim to visualize this data to help communicate the data distribution. In doing this we also create a copy of the dataset to maintain its integrity for modifications. We calculate the distribution first with `df_cleaned['Guild_Membership_encoded']` extracts the column representing the guild membership and `.value_counts()` counts each unique value in the column. We then print out the count and represent it visualizing using a bar graph through, `distribution.plot(kind='bar', …)`.
The outputs show the precise distribution in the column. The majority of rows in the group (59,991) belong to category 0 (`No_Guild`). The second majority of the rows (31,671) belong to category 1 (`Master_Guild`) and lastly, category 2(`Apprentice_Guild`) has the smallest portion of the cleaned dataset (4,166).  This distribution represents a significant imbalance in the dataset which will be considered in the later analyses and modeling to avoid inaccurate and biased results.

#### Part V of Step 2: Data Splitting Amongst Categorical and Numerical Columns
* 2.5.1) Then we reduce the no guild and master guild columns to 20000 for no guild and 13000 for apprentice guild. And with this it creates new dataset called `df_balanced`
As there is a class imbalance in our dataset, we implemented a resampling strategy to address this problem. We downsample the over-represented classes (categories 0, and 1) and keep the minority class (category 2) unchanged. In the code, we split the data into the respective classes. We then proceed to reduce the over-represented classes with:
- `class_0_downsampled = resample(class_0, replace=False, n_samples=20000, random_state=42)`
- `class_1_downsampled = resample(class_1, replace=False, n_samples=13000, random_state=42)`
* …while the minority class is kept unchanged with:  `class_2_unchanged = class_2`. 
* We then combine the downsampled subsets and unchanged class into a single DataFrame (`df_balanced`). We save the reduced Dataframe as a new CSV file to be referenced later, again to protect each changed CSV from the previous one. Lastly, we perform the counting of distributions and the visualisation of the distributions utilized in the previous step L to output the updated class distribution figures. The result of this method is the reduced dataset set is **37,166 rows** (*20,000 (class 0) + 13,000 (class 1) + 4,166 (class 2)*). In the end, there is an improvement in model fairness as the models that will be trained with this dataset are less likely to favour the dominant class and are better equipped to learn the patterns in the smaller classes.
* 2.5.2) Splitting the columns into categorical and numerical in the new dataset to prepare them for filling in the missing values using KNN. The initial step is to identify the categorical and numerical columns in the dataset. Categorical columns were selected based on their data types: *object* for strings and *bool* for binary data. These columns will be encoded before using KNN, as KNN works with numerical data. Numerical columns were identified by *float64* for continuous data and *int64’* for integers. The columns will be used directly after normalization. We further split the numerical columns into `discrete_columns = []` and `continuous_columns = []`. For handling missing values, once split, discrete columns can be imputed with KNN, treating them as integer-like features, while continuous columns can be imputed with KNN, leveraging their continuous nature for distance calculations. Discrete numerical columns are columns with fewer than 20 unique values, as determined previously, and continuous numerical columns are columns with more than 20 unique values. We then display an output of the split categories as a list. This method provides proper feature preparation which enhances the performance of KNN by ensuring meaningful distance calculations.
* 2.5.3) Doing KNN first for numerical continuous and discrete and THEN encoding the categorical values utilizing custom mapping to do KNN for categorical columns. 
	**KNN imputation for numerical columns:**  
		**KNN imputing for continuous features**: We start by scaling the continuous features with the use of `StandardScaler()` to standardize the where mean equals 0 and variance is equal to 1. Scaling is done to ensure features with larger ranges do not dominate the KNN distance calculations. 
		**KNN imputation of discrete columns**: This handles the missing values which replace missing values based on the mean of the k nearest neighbors in the feature space. KNN imputation is done differently as these are directly imputed without scaling. Imputed values are rounded and converted back to integers to preserve their discrete nature.
		**KNN imputation for categorical variables:**
Firstly, we *inspect the categorical columns* by displaying unique values for each column to understand the range of possible values; done with `f"{col}: {df_balanced[col].unique()}".` This helps in preparing for the next steps, which are encoding and imputation
As aforementioned, encoding is a necessary step in imputing categorical variables as KNN works with numerical data. Our encoding function first cleaned the values by converting all values to lowercase and removing the extra whitespaces. Then, it maps categorical values to binary `df_encoded[col].map({'yes': 1, 'present': 1, 'no': 0, 'absent': 0})`. The final step of the encoding function is handling missing values by replacing them with -1 as a placeholder.
We finally go ahead with the KNN imputation which is the same process as the numerical importing above. The process is done by guaranteeing imputed values remain binary after rounding: `‘imputed_binary = imputed_binary.round().astype(int) df_copy[binary_columns] = imputed_binary`
After each step, we verify and validate that the missing values have been successfully imputed. This is for us to check the effectiveness and success of the imputation
**Updating column lists:** We have to ensure that the dataset is complete and ready for the step while maintaining the integrity of each feature type.
We clean and ensure consistency by `list(numerical_columns)` and `list(set(categorical_columns))` are unique and mutually exclusive.
We adjust the lists by moving the target column, `Guild_Membership_encoded`, to the numerical/ categorical columns if necessary
As the final step, we displayed the updated column lists for final verification.

#### Part VI of Step 2: Identifying correlations and distributions of relevant features for categorization
* 2.6.1) Checking for outliers and handling them in a way that the once that are on above the plot distribution  we set as the highest value within the plot and the once under we set as the lowest value within the plot.
Before checking for outliers we created a copy of the dataset to only show the outliers but then we proceeded to alter the `df_balanced` when handling them. The handling methods involve capping outliers to the nearest acceptable values based on the IQR.

<img width="1242" alt="Screenshot 2024-12-03 at 23 37 55" src="https://github.com/user-attachments/assets/d0301a9b-45cc-460e-a943-c064058898cd">
Left (outlier detection), Right (outlier handling)

**Outlier Detection** 
We calculated the outliers by defining the values that are not within the boundaries of lower (Q1 - 1.5 * IQR) and upper bounds (Q3 + 1.5 * IQR).  Once calculated for each column, it is printed and plotted as boxplots: 

The outliers of each category are as follows, `Fae_Dust_Reserve` has 745 outliers, `Physical_Stamina` has 5372 outliers, `Mystical_Index` has 772 outliers, `Mystic_Energy_Level` has 1456 outliers, `Age_of_Wisdom` has 1069 outliers, `Mental_Wizardry` has 4528 outliers, `General_Health_Condition` has 2020 outliers, `Knightly_Valor` has 50 outliers, and `Gold_Pouches_Per_Year` and `Level_of_Academic_Wisdom` both have 0 outliers. 
As seen there is a large variability and there is a presence of extreme values in some of the data, resulting in those outliers.
**Outlier Handling (Capping)** 
We replace outlier values with the nearest acceptable bound to ensure extreme values are within a valid range. `df_balanced[column] = df_balanced[column].clip(lower=lower_bound, upper=upper_bound)` Values below the lower bound are set to the lower bound and the values higher than the upper bound are set to the upper bound. The result after capping there are no outliers


* 2.6.2)Revisiting all the results from our myriad of cleaning and preprocessing. Calculating the correlation of features with the target variable, `Guild_Membership_encoded`, and visualizing the updated distributions of each feature across different guild membership categories.
We perform a correlation analysis on the target variable where we discover the most positively correlated features are `General_Health_Condition` and `Royal_family_pressure_Presence`, and the most negatively correlated features are `Level_of_Academic_Wisdom` and `Gold_Pouches_Per_Year`. The positively correlated feature may indicate a significant role in distinguishing guild membership categories while the negatively correlated may indicate a potential inverse relationship with guild membership categories. Additionally, the features with (almost) no correlation, eg. `Knightly_Valor` with a value of **-0.000486**, may indicate that these values might be less relevant for predictive modelling.
**Feature Distribution Visualization**
We visualize the distribution of each feature in the dataset (`df_balanced`) across the three categories (1, 2, 3) of the target variable.
*Histograms* `sns.histplot`: Plots histograms for each feature, breaking down their distributions across the categories of `Guild_Membership_encoded` using the hue parameter. Each category has its own separate bar to make their distribution visually distinct. 
*Output* 
* Features with Distinct Separations: Features like `General_Health_Condition`, `Royal_family_pressure_Presence`, and `High_willingness_Presence` show noticeable differences in distribution between guild membership categories. For example: `The Royal_family_pressure_Presence` and `High_willingness_Presence` distributions suggest these traits are more prominent for specific guild categories.
* Features with Similar Distributions: Features such as `Mystic_Energy_Level` and `Knightly_Valor` appear to have overlapping distributions across guild membership categories. These might not provide much discriminatory power for classification.
Skewed Features: Certain features, like `Gold_Pouches_Per_Year` and `Level_of_Academic_Wisdom`, exhibit skewed distributions. `Gold_Pouches_Per_Year` likely shows a right-skewed distribution, as most guild members/ the population have low or moderate gold reserves, with very few having extremely high values.
* 2.6.3) Plotting correlation heatmaps for each unique value within the Guild Membership column (0,1,2) to show how each feature correlates to the classification of each respective scholar
We generate a heatmap for each guild membership type with the `unique()` method. Firstly, we make a subset `guild_data` to include only rows corresponding to the current `guild` category which is used to calculate correlations.
<img width="1343" alt="Screenshot 2024-12-03 at 23 34 51" src="https://github.com/user-attachments/assets/20602f0d-0696-4e9f-a5ac-19cf50d6550e">

Most Relevant Features for Classification

From **No Guild (0)** to **Master Guild (1)**:
1. Defense_spell_difficulty_Presence
2. General_Health_Condition
3. Physical_Stamina
4. Royal_family_pressure_Presence

From **No Guild (0)** to **Apprentice Guild (2)**:
1. Mystical_Index
2. Mystic_Energy_Level
3. Level_of_Academic_Wisdom
4. General_Health_Condition


### 3) Model Preparation
* 3.1) Once we’ve collected all our data, Splitting our data into testing and training sets, 20% assigned to test set (With more training samples, the model is more likely to encounter variations in the data, making it better at generalizing to unseen cases) (80% allocated for training ensures the model has sufficient data to learn patterns and relationships among features and target classes) -> prints shape of data 

	We use the 80-20 split which provides a balanced approach:
 -  **80% training data:** Maximizes learning potential for the model.
 - **20% test data:** Ensures robust evaluation of the model's performance and generalization ability.
A smaller training set (e.g., 70-30 split) might lead to underfitting due to insufficient data for training. Conversely, a smaller test set (e.g., 90-10 split) may not provide enough data for reliable evaluation.

We implement the split by first defining the input features: **X:** The feature matrix containing independent variables, **y:** the target variable containing class labels. Then, we specify that 20% of the data should go into the test set. We include `random_state=42`to ensure reproducibility by fixing the random seed, and `stratify=y` to maintain the portion of target classes in the training and testing sets.
1. **Feature Matrix**: `X_train_final shape: (19945, 20), X_val shape: (4987, 20)`. Both subset have the same feature count proving consistency for model training.
2. **Target Labels**: `y_train_final shape: (19945,), y_val shape: (4987,)`. The use of stratified sampling assures that the class distribution in `y_train` and `y_test` mirrors the original distribution in y. This is important for potentially imbalanced datasets like ours to prevent biased splits.

* 3.2) We scale the training and testing tests to standardize features to ensure each feature contributes equally to the model. We do the scaling to a copy of the test and training sets for security. This can only be performed on numerical columns for their quantitative characteristics.
With a list comprehension for loop, we identify and extract the numerical columns (with data types `float64` or `int64`) from `feature_columns`. Scaling is done only with numerical columns because applying it to categorical columns can distort their value and meanings. 

 - `fit_transform` calculates the mean and standard deviation of each numerical feature in `X_train` and scales the data accordingly. `transform` applies the scaling parameters learned from `X_train` to `X_test`.
 - **Output:** There is an equal contribution from features so, for example, features like Gold_Pouches_Per_Year do not dominate smaller-scale features. There is also consistency in the scaling due to the testing set scale using parameters from the training set evidenced by `Fae_Dust_Reserve`: Training (-1.162043 to -0.980271), Testing (-0.268081 to 1.051221).
* 3.3) Further splitting the scaled training set dataset for the final dataset and validation set that remains independent of the training process.
As above in 3.1, we split the data into 20-80% of training and testing. The parameters are `X_train_scaled` and `y_train`: The scaled features and corresponding labels from the initial training set. We again stratify `y`. We get the final training and validation set where there are 19,945 samples in the final training set and 20 columns. The feature matrix of the validation set has 4,987 samples, which is 20% of the original training set and 20 columns. The output shows a proportional split applied to the training set and consistency between the training and validation sets, which is critical for training ML models.


### 4) Training and Testing Models 
* 4.1) Through anticipated research in preparation for this Royal Task, we determined the 4 most adequate models to complete a multi-class classification task, like our own. This led us on a vast journey, and we came to a conclusion of testing the following 4:
	* 1) Gradient Boosted Trees
	* 2) Random Forest
	* 3) Logistic Regression
	* 4) K-Nearest Neighbors (KNN) - with 5 neighbors
The models are set in their own dictionary and we set a dictionary to store our results. 
We establish a for loop and iterate over each key and value in the dictionary to train our four models. This simultaneously calculates a series of metrics.
* Our output, for each model and its performance, is sectioned into 7 intersecting metrics:
	`**Accuracy:**`
		- Proportion of correctly predicted labels out of the total predictions.
		- **Formula**: Accuracy = $\frac{Correct Predictions}{Total Predictions}$ 
		- Represents overall performance but may not reflect model performance well in imbalanced datasets.
	`**Precision:**` How many of the predicted positive results are correct?
		**Formula**: Precision =  $\frac{True Positives}{True Positives + False Positives }$ The higher the precision, the fewer false positives predicted.
	`**Recall (Sensitivity):**` How many of the real positive results were correctly identified.?
		Formula: Recall= $\frac{True Positives}{True Positives + False Negatives }$ The higher the recall, the fewer false negatives predicted.
	`**F1-Score:**` Calculates the harmonic mean of precision & recall.
		Formula: F1=2 ×  $\frac{Precision × Recall}{Precision + Recall}$ 
		When handling imbalanced datasets, F1 score helps to balance both metrics of assessment to provide a more calibrated result.. 
	`**Support:**` The number of occurrences of each Guild Membership (3)  in the data.
	`**Macro Average:**` Calculates the unweighted average of: precision, recall, & F1-score across all Guilds, testing to balance the significance of each class for comparison
	`**Weighted Average:**`
	Averages: precision, recall, & F1-score, this time taking into account the weights of each individual Guild.

* **MODEL EVALUATION FROM FIRST RUN**:
	- **1) Gradient Boosted Trees**:
		- **Accuracy**: 65.6%
		- **F1 Score – Based on Guild Membership**: 
			* **No Guild**: 76%
			* **Master Guild**: 60%
			* **Apprentice Guild**: 1%
		- **Macro Average**: 46%
In terms of accuracy and weighted F1 score, this model performs the best overall. The model performs well for classifying the `No Guild` Scholars and `Master Guild` Scholars; however, it should be noted that the model heavily struggles with the `Apprentice Guild` class, along with the other models. GBT is successful at understanding complex dataset patterns but is highly receptive to imbalances in class, as seen within the individual Guild Membership performance. This type of model is much more suited for well-established and balanced classes.
	- **2) Random Forest**:
		- **Accuracy**: 64.2%
		- **F1 Score – Based on Guild Membership**:
			* **No Guild**: 75%
			* **Master Guild**: 58% 
			* **Apprentice Guild**: 2%
		- **Macro Average**: 45%
Random Forest performed slightly below GBT. It performed strongly for classifying the `No Guild` Scholars, and similarly for the `Master Guild`; however, struggles with the `Apprentice Guild`, similarly to Gradient Boosted Trees. In terms of Macro Average, Random Forests has a lower performance at 45%. This classifier works well with an imbalance dataset, like our own, but does not possess the boosting power that GBT has, though it is much quicker in terms of computation time.
	- **1) Logistic Regression**:
		- **Accuracy**: 64.6%
		- **F1 Score – Based on Guild Membership**:
			* **No Guild**: 75%
			* **Master Guild**: 58%
			* **Apprentice Guild**: 0%
		- **Macro Average**: 45%
Logistic Regression is much less granular and complex, and computes quickly. However, let it be known, that because it is so simple, it fails to classify the `Apprentice Guild` **at all**. Its overall precision, sensitivity, and F1 Scores for this class are all 0. As a model, Logistic Regression is linear and thus it poorly handles complex relationships and non-linear presences in datasets. This makes the classifier extremely sensitive the imbalance Guild Membership and needs tuning to improve its accuracy and performance.
	- **1) K-Nearest Neighbors**:
		- **Accuracy**: 60.1%
		- **F1 Score – Based on Guild Membership**:
			* **No Guild**: 73%
			* **Master Guild**: 51%  
			* **Apprentice Guild**: 6%
		- **Macro Average**: 45%
We immediately notice that of all accuracies, the KNN Classifier performs more poorly than the other 3 models. KNN as a classifier is dependent mostly on the parameters that have been chosen to tune its performance, because of this, it scales with struggle and cannot handle the dataset’s imbalance. This motivates us to drop this model from further consideration as a contender for hyperparameter tuning. KNN tends to overfit the training data and is much better suited for smaller, more elementary datasets. 

* Though we have heavily reduced our dataset, it is clear that the models still suffer from the “imbalance” present. The excelling performance in the `No Guild` classifications prove that the amount of data this Guild has, in comparison with the rest, is dominating the training and testing. We notice that Gradient Boosting Trees, Random Forest, and Logistic Regression *outperform* KNN; GBT and Random Forest are more complex in nature, and Logistic Regression offers a competitive speed and simplicity. We will avoid KNN for its poor performance and vulnerability to the dataset’s dimensionality, and continue focusing on the **remaining 3 contenders:**
- `Gradient Boosted Trees`
- `Random Forest`
- `Logistic Regression`


### 5) Plotting our Learning Curves

* 5.1) Along with training and testing our 4 selected models, it is crucial to spatially conceptualize the performance of these models. We accomplish this by plotting Learning Curves, with the `.plot_learning_curve( )` built-in function. We define our proportions of training sizes, that eventually grow from 10-100%, to plot the Learning Curves. We initialize the training and validation score lists for the varying training sizes, and iterate our training over this range. By this process, we are able to calculate the training **accuracy** and validation **accuracy** with a `.score( )` built-in function. These accuracies are plotted with the growing training set sizes, allowing us to visually perceive the behavior of each selected model, and understand how the models perform, relative to our split set proportions.

* 5.2) We observe the following from our Learning Curves, assorted by model type:
   - **1) Gradient Boosted Trees:**
      * **Training accuracy**: scores initially perform well, resulting with a score upwards of 76%, however, as the training set size begins to grow in proportion, the accuracy drastically decreases and levels out around the 80% training set size to a mid-late 60s percentage of accuracy
      * **Validation accuracy**: has a seemingly opposite trend, mirroring the training accuracy, but on a less dramatic scale. The accuracy begins below 62% and rises to nearly 66%, eventually plateauing at that accuracy score.
      * **Analysis:** *LET IT BE KNOWN*: This model has a significantly longer runtime, deeming it inefficient in comparison, in terms of computation time and execution. This completely impacts our model selection for future hyperparameter tuning.![14ea4abf-8897-4e2f-ad8c-74b1009f1a74](https://github.com/user-attachments/assets/5a4bbf3a-5b76-4964-9e13-d8f9dc1024df)


   - **2) Random Forest:**
      * **Training accuracy**: here begins at an extreme accuracy of 100%, and remains there
      * **Validation accuracy**: begins inaccurately around 63% and proceeds to remain about that percentage.
      * **Analysis**: These results are clearly insinuating that the model is **highly overfitting** the data and outputting skewed accuracies for the training sets, a matter that must be fixed for proper evaluation.![258f5df8-c997-410d-a1f1-b86c75328dec](https://github.com/user-attachments/assets/5633b212-8b3d-4ee4-b1b6-16ca8de3e9b2)


   - **3) Logistic Regression:**
      * **Training accuracy**: here commences at 64.4% and reaches a minimal high point at 65.1% around a 70% training set size proportion. Because this variation is minimal, the Learning Curve is plotted on a much smaller grid size, with smaller units. 
      * **Validation accuracy**: begins lower, at about 64.1% accuracy, and reaches a maximum height at 64.6% accuracy when the training set proportion is at 100%.
      * **Analysis**:  Both of these accuracies remain relatively low and indicate that the model might be too superficial to actually contextualize the complex dataset (underfitting). Though the gap between accuracies is much smaller than the first two, indicating success with overall generalization, the model still doesn’t perform well enough for this kind of task.![ce9fab08-0b95-4a91-a5db-b241324508b6](https://github.com/user-attachments/assets/284a3222-c310-4c12-8871-def21eed36a5)

  
  - **4) K-Nearest Neighbors:**
     * **Training accuracy**: Begins at about 70% and remains consistent here as the training set proportion grows.
     * **Validation accuracy**: begins lower at about 60%, and fluctuates,, in slight as the proportion of training set size increases, but only peaks at 62% and declines back down to 60.5%
     * **Analysis**:  There is a notable gap between these two accuracies, and this leads to an assumption that this model overfits the given data. Though the model can properly memorize the data of the training set, it has complications generalizing the yet-to-be-seen validation data. 
![2acbbc1e-b155-495c-b7be-2f97a9888ea3](https://github.com/user-attachments/assets/9ad00ac6-e2f9-4211-82cf-a419603db681)

**INSIGHT**: 
* Here, we focus on understanding how the models perform relative to our split set proportions. The learning curves display how the training/validation accuracies fluctuate as the training set size grows. These plots are very indicative for our models, explaining how exactly our data size affects our models and their ability to perform on our designated data.
	* **Gradient Boosted Trees** maintains a balance between generalization & performance, making it the best candidate thus far, 
	* **Random Forest** is very susceptible to overfitting without tuning/regularization.
	* **Logistic Regression** is too far too linear/simple and underfits the data, though it shows strong generalization.
	* **KNN** also overfits and poorly generalizes along with its low performing accuracy


### 6) Evaluating our Models:
* #### **Attempt 1 **  
* We chose to hyperparameter tune the following models: **Gradient Boosting Trees** (initially the best performer), **Random Forest** and **Logistic Regression**. 
* During our first attempt to perform hyperparameter tuning on these models, we initially attempted to optimize Gradient Boosting Trees due to its promising accuracy during initial testing. However, in the tuning process we encountered significant challenges with the computational efficiency of performing **Grid Search** on **Gradient Boosting Trees**. Despite experimenting with various adjustments to lower the time it takes for the tuning process, the model's training time remained too long (shown in the screenshot below). Given that computational efficiency is a critical consideration in real-world applications, we made the decision to exclude Gradient Boosting Trees from further optimization. This allowed us to focus on the two remaining models, **Logistic Regression** and **Random Forest**, which demonstrated balanced performance and acceptable computational efficiency during the initial evaluation.This decision ensured that our workflow remained efficient while concentrating on models with a better balance of performance and practicality.

* #### **Attempt 2:**  
Following the initial training phase, we prioritized hyperparameter tuning for Logistic Regression and Random Forest. These models were selected due to their potential for improvement, coupled with their computational practicality, making them better suited for further refinement and optimization in this project. 

* Another issue we encountered after the initial training of the models was their inability to effectively classify the minority class, **Apprentice Guild**. To address this, we decided to slightly oversample this class by adding 1,000 additional samples just before performing hyperparameter tuning on the models.


### 7.1) Problem Solving, Pre-Tune:
#### **Attempt 1:**  
We chose to hyperparameter tune the following models: **Gradient Boosting Trees** (initially the best performer), **Random Forest** and **Logistic Regression**. 
During our first attempt to perform hyperparameter tuning on these models, we initially attempted to optimize Gradient Boosting Trees due to its promising accuracy during initial testing. However, in the tuning process we encountered significant challenges with the computational efficiency of performing **Grid Search** on **Gradient Boosting Trees**. Despite experimenting with various adjustments to lower the time it takes for the tuning process, the model's training time remained too long (shown in the screenshot below). Given that computational efficiency is a critical consideration in real-world applications, we made the decision to exclude Gradient Boosting Trees from further optimization. This allowed us to focus on the two remaining models, **Logistic Regression** and **Random Forest**, which demonstrated balanced performance and acceptable computational efficiency during the initial evaluation.This decision ensured that our workflow remained efficient while concentrating on models with a better balance of performance and practicality.
<img width="566" alt="Screenshot 2024-12-03 at 22 55 45" src="https://github.com/user-attachments/assets/ab17b54d-24fe-43b1-a4a8-53dcee1b8c4a">

#### **Attempt 2:**  
Following the initial training phase, we prioritized hyperparameter tuning for Logistic Regression and Random Forest. These models were selected due to their potential for improvement, coupled with their computational practicality, making them better suited for further refinement and optimization in this project. 

Another issue we encountered after the initial training of the models was their inability to effectively classify the minority class, **Apprentice Guild**. To address this, we decided to slightly oversample this class by adding 1,000 additional samples just before performing hyperparameter tuning on the models.

### 7.2) Hyperparameter Tuning
####Logistic Regression
We chose this model due to its simplicity, computational efficiency, and stable performance during initial testing. The tuning process aimed to optimize:
**Regularization Strength (`C`)**: Adjusted to find the optimal trade-off between underfitting and overfitting. 
**Solver**: Explored options like `lbfgs` and `saga` to handle the multi-class nature of the problem effectively. 
**Maximum Iterations (`max_iter`)**: Increased to ensure convergence for the high-dimensional dataset.
**RESULTS:**
**The best parameters obtained: **  `C: 10`, `solver: lbfgs`, `max_iter: 500`
**Cross-validation accuracy**: Improved to **62.46%**, a slight increase from the initial performance. 
**Training time**: Highly efficient (~4.49 seconds)
**Analysis**: The optimized Logistic Regression model maintained its computational efficiency while achieving a modest improvement in performance. 

A.1 **Warnings in Logistic Regression Hyperparameter Tuning**
During the hyperparameter tuning phase for Logistic Regression, we encountered `ConvergenceWarning` messages. These warnings typically indicate that the optimization process did not fully converge within the specified maximum number of iterations (`max_iter`).
The warnings are triggered during the exploration of certain parameter combinations in the grid, particularly those involving high regularization (`C`) or specific solvers like `saga`. These combinations require more iterations to converge due to the complexity of the optimization problem. 
**Impact on Results: ** The presence of `ConvergenceWarning` during hyperparameter tuning does not invalidate the results:
The best parameter combination (`C=10`, `solver=lbfgs`, `max_iter=500`) was selected after careful cross-validation and is unaffected by these warnings. 
The warnings are an indication of computational inefficiency in exploring specific parameter combinations rather than errors in the tuning process.


#### Random Forest
This model demonstrated strong potential during initial testing but exhibited signs of overfitting, which is indicated by a significant gap between training and validation accuracy. By hyperparameter tuning this model we aimed to improve its generalization by optimizing the following parameters: 
**Number of Trees (`n_estimators`)**: Adjusted to balance accuracy and runtime. 
**Tree Depth (`max_depth`)**: Limited to prevent overfitting. 
**Minimum Samples per Split (`min_samples_split`)** and **Leaf (`min_samples_leaf`)**: Tuned to optimize the tree-splitting process. 
**Features Considered per Split (`max_features`)**: Explored to enhance feature selection at each node.
**RESULTS:**
**Initial Randomized Search**: 
**Best Parameters**: `n_estimators: 100`, `max_depth: 20`, `min_samples_split: 5`, `min_samples_leaf: 1`, `max_features: log2` 
**Cross-validation accuracy**: Improved to **70.93%**. 
**Training time**: ~74.69 seconds.
**Final Grid Search**:
**Best Parameters**: `n_estimators: 150`, `max_depth: None`, `min_samples_split: 4`, `min_samples_leaf: 1`, `max_features: log2` 
**Cross-validation accuracy**: Further improved to **71.22%**. 
**Training time**: ~107.93 seconds.
**Analysis**: The tuning process significantly improved Random Forest’s performance, reducing overfitting and enhancing its ability to generalize. By comparing the Randomized Search and the Grid Search we notice that the combination of increased tree depth and fine-tuned splitting parameters allowed the model to better capture the dataset's complexities. However, the computational cost of the final grid search was notably higher than the randomized search, highlighting a trade-off between performance and runtime. 

We use these optimized parameters for further model training and evaluation.


### 8) Evaluating our Models:
* 8.1 We tested the performance of the models—Random Forest and Logistic Regression—using the best parameters identified from previous hyperparameter tuning. Along with the already-visualized accuracy, precision, recall, and F1 score, we also plotted a confusion matrix, with the class predictions in colour of intensity. 
We define the two models with `models_with_best_params` in which the models are then trained with the full training sets (`X_train_final` and `y_train_final`). The accuracy and runtime are calculated on the validation set `y_val_pred` and the classification reports are printed out which include precision, recall, and F1 scores for each class. The A function `plot_learning_curve_best` evaluates model performance with varying amounts of training data.
#### **Random Forest Results**
- **Accuracy:** 0.7284
- **Class-Level Performance:**
 	 - **Class 0 (No Guild):** High precision (0.75) and recall (0.81), indicating strong performance for this majority class.
  - **Class 1 (Apprentice):** Moderate precision (0.61) and recall (0.63), showing decent but not perfect predictions for this group.
 	 - **Class 2 (Master):** Very high precision (0.98), but lower recall (0.65), meaning the model predicts "Master" accurately but misses some actual instances.
- **Macro Average F1-Score:** 0.73
- **Weighted Average F1-Score:** 0.73
- **Training Time:** 3.15 seconds
- **Generalization:** The model does not show signs of overfitting or underfitting, with validation performance aligning closely with expected outcomes.

#### **Logistic Regression Results**
- **Accuracy:** 0.6206
- **Class-Level Performance:**
 	 - **Class 0:** Moderate precision (0.67) and high recall (0.82), showing a tendency to favor this majority class.
  	- **Class 1:** Moderate precision (0.53) and recall (0.61), similar to Random Forest but slightly worse overall.
 	 - **Class 2:** Completely fails to predict this class, with precision, recall, and F1-score at 0.00.
- **Macro Average F1-Score:** 0.44
- **Weighted Average F1-Score:** 0.56
- **Training Time:** 0.30 seconds
- **Generalization:** Logistic Regression underfits the data, struggling to capture the complexity of class distributions, particularly for the "Master" class.

#### **Comparison and Evaluation**
- **Accuracy:** Random Forest outperforms Logistic Regression, showing better generalization across all classes.
- **Class-Level Performance:**
  - Random Forest handles all classes reasonably well, including the "Master" class, with a high precision of 0.98. Logistic Regression, however, fails entirely for the "Master" class.
  - Both models perform moderately for the "Apprentice" class, but Random Forest achieves better precision and recall.
  - Logistic Regression over-prioritizes the majority "No Guild" class, which impacts overall performance.
- **Macro and Weighted F1-Scores:** Random Forest has significantly better F1-scores (0.73 vs. 0.44 macro and 0.73 vs. 0.56 weighted), reflecting its balanced performance across all classes.

-  We have concluded that Random Forest is the superior model for this dataset, offering better accuracy, class-level performance, and generalization. Logistic Regression may still be useful for simple and linear tasks or when computational efficiency is a priority, but it is not suitable for this classification task.

* 8.2 For a final model evaluation, we calculate the ROC Curves and AUC scores for our multi-class classification task, approached with “one-v-all”. As well, we compute a confusion matrix `ConfusionMatrixDisplay` for each model to highlight potential misclassifications. We compute macro averaging, used to equally weight the performance of each class, irregarding the class proportions; a very common multi-class classification strategy. For each class, we compute the ROC Curve and calculate the AUC as if it were a binary problem, and later take the simple average of AUC scores for all classes to compute the macro-averaged AUC. 
#### Random Forest Results
- **`Macro-AUC`: 0.8448:**This indicates that the Random Forest model performs well across all classes, with strong discriminative power.
- **Confusion Matrix Observations:**
	- *Class 0:* The model correctly classified the majority of instances (2204 correct, 510 misclassified as Class 1).
	- *Class 1:* Moderate performance with 1014 correct predictions but 582 misclassified as Class 0.
	- *Class 2:* Reasonable performance, with 560 correct classifications but noticeable confusion with other classes.
- This final evaluation shows that Random Forest handles all three classes reasonably well, with the highest performance for Class 0 which is the No Guild category.

#### Logistic Regression Results
- **`Macro-AUC`: 0.7337**:This reflects how Logistic regression has a lower performance compared to Random Forest.
- **Confusion Matrix Observations:**
	- *Class 0:* Good performance with 2236 correct predictions, though some were misclassified as Class 1.
	- *Class 1:* Moderate results, with 983 correctly predicted but a higher rate of confusion with Class 0.
	- *Class 2:* The model was not very successful to predict Class 2, with all instances misclassified as either Class 0 or Class 1.
- Lastly, Logistic Regression struggled significantly with the minority class (Class 2), leading to lower overall performance. It especially has lower performance than random forest.

#### Graph Observations
<img width="803" alt="Screenshot 2024-12-03 at 23 28 05" src="https://github.com/user-attachments/assets/7143f1dd-e188-47eb-abe4-d819443a6fe1">

1. ##### Random Forest
The curve is closer to the top-left corner of the plot, indicating high sensitivity (**True Positive Rate**) at lower False Positive Rates.

2. ##### Logistic Regression
The curve is less steep and farther from the top-left corner compared to Random Forest, indicating lower sensitivity at equivalent False Positive Rates.

### 9) Hyperparameter Sensitivity Analysis:
It is paramount that we visualize the impact of the hyperparameters on our Random Forest Classifier’s training and test scores with line plots. This is completed by finding the mean scores for varying values of the number of trees in the forest (`n_estimators`), the maximum depth of the trees (`max_depth`), (`min_samples_split`) the minimum number of samples required to split an internal node, (`min_samples_leaf`) the minimum number of samples required to be in a leaf node, and (`max_features`) the maximum number of features considered for splitting a node. This step allows us to see how increasing the number of trees or adjusting the depth of the trees affects model performance (under vs. overfitting), The key hyperparameters are derived from the cross-examination portion of hyperparameter tuning of Random Forest via Randomized and Grid search. For the code, we use Randomized search to see how the ideal parameters for Gridsearch were chosen, then we see the fine-tuning of Gridsearch in the parameters.
<img width="738" alt="Screenshot 2024-12-03 at 23 28 28" src="https://github.com/user-attachments/assets/52832653-203c-4d66-8c66-a359a0738b2e">

#### **Randomized Search**
- `n_estimators`: 
    - Training Score: remains consistently high, around 85%-95%, for all values of `n_estimators`.
    - Test Score: remains lower than the training score, indicating a possible generalization gap.
- `max_depth`: The maximum depth of each tree.
    - Training Score:As shown, the score increases with `max_depth`, reaching nearly 100% at `max_depth = 20`, and then slightly declines as the model becomes overfitted or overly complex.
    - Test Score:it peaks at around `max_depth = 20`, where the model achieves the best balance of complexity and generalization.
- `min_samples_split`:
    - Training Score: The training accuracy decreases slightly as min_samples_split increases. This is expected since larger splits make the model less complex by preventing small, specialized splits.
    - Test Score: The test accuracy remains relatively stable, slightly peaking at lower `min_samples_split` values (around 2-5) and then plateauing or slightly decreasing. This suggests that smaller split thresholds provide sufficient complexity without overfitting.
- `min_samples_leaf`:
    - Training Score: The training accuracy peaks at `min_samples_leaf = 2` and starts to decline with larger leaf sizes. Smaller leaves allow the model to fit the training data more closely, increasing accuracy but risking overfitting.
    - Test Score: The test accuracy mirrors this trend, with a slight peak at min_samples_leaf = 2 and a decline thereafter. Larger leaves simplify the model too much, leading to underfitting.
- `max_features`:
    - Training Score: The training accuracy increases as the number of features considered for splits grows (`sqrt to log2`). This happens because more features allow the model to fit the training data more precisely.
    - Test Score: Test accuracy also improves as max_features increases, with the model achieving the best balance at `log2`. This suggests that considering more features helps the model generalize better without overfitting.

<img width="738" alt="Screenshot 2024-12-03 at 23 28 32" src="https://github.com/user-attachments/assets/915af8d5-91c4-421e-a067-cd0fd806bc73">

#### **Grid search**
- `n_estimators`: 
    - Training Score: Remains consistently high, around 90%-95%, across all values of `n_estimators`. This indicates that the model is sufficiently complex to fit the training data well.
    - Test Score: Remains lower than the training score, in the range of 65%-70%. This highlights a potential generalization gap, where the model may be overfitting.
- `max_depth`:
    - Training Score: Increases with `max_depth`, reaching nearly 100% at `max_depth = 20`. This reflects that deeper trees can perfectly fit the training data.
    - Test Score: Increase as approaches the around `max_depth = 20`, where the model achieves the best balance between complexity and generalization. Beyond this point, overfitting may cause the test performance to degrade.
- `min_samples_split`: 
    - Training Score: Decreases slightly as `min_samples_split` increases from 4 to 6. This indicates that higher values of this parameter make the model less flexible, which plays in reducing its ability to fit the training data perfectly.
    - Test Score: Remains relatively stable around 65%-70%, suggesting this parameter has minimal impact on the model’s generalization ability.
- `min_samples_leaf`:
    - Training Score: Decreases as `min_samples_leaf` increases from 1 to 2. Larger leaf sizes enforce a simpler model, reducing overfitting.
    - Test Score: Remains relatively stable at approximately 65%-70%, indicating robustness to changes in this parameter.
- `max_features`: 
    - Training Score: Very high (around 90%-95%) with the derived feature option from Randomized Search, `log2`.
    - Test Score: Lower than the training score, approximately 65%-70%, with a notable generalization gap. This suggests careful tuning of max_features might be needed to improve test accuracy.

#### Conclusion
- After conducting a randomized search for initial hyperparameter exploration, a grid search was performed to fine-tune the selected parameters with cross-validation. The results indicate that `n_estimators` and `max_depth` play a significant role, with training scores consistently high and test scores peaking at balanced values. The rest of the parameters like `min_samples_split`, `min_samples_leaf`, and `max_features` show stable test performance, with optimal values preventing overfitting or underfitting, demonstrating the model's sensitivity to complexity control.
- The sensitivity analysis provides strong evidence to support the hyperparameter choices from the hyperparameter tuning we have done earlier. These visualizations not only validate the optimal values but also highlight how performance would degrade with less optimal configurations. This alignment increases our confidence in the tuned Random Forest model's robustness and generalization capabilities.

### Which is the **Best Model** to Complete the Royal Task?
 Your excellency, after our tireless preparation, training, and evaluation efforts, we– your humble workers, have deduced that the **Random Forest** model is perfect for this Royal Task. In search of the optimal model, we have brought forth four machine learning model options: Gradient Boosted Tress, K-Nearest Numbers, Logistic Regression, and Random Forest. We have trained and validated these models with a dataset we have refined to make the most accurate, unbiased, and precise predictions. After the initial training and testing, we have found out that in descending order the ones with the highest validation accuracy: GBTs, Logistic Regression, Random Forest, and lastly KNN. Even though GBT had the highest accuracy we have moved forth with hyperparameter tuning Random Forests and Logistic Regression as they were significantly more computationally efficient when compared to GBTs. Through our efforts of hyperparameter tune with the cross-examination methods we evaluated our refined models. We have found out that after extensive evaluations, though logistic regression had great accuracy, random forests dominated in predicting ability and accuracy in comparison. Thus, mi-lord, we have decided the Random Forest is the best model for your Highness’ Royal Task.

### 10) Our Expert Cut: How Can We Optimize Guild Prediction?
We’ve trekked through our long, vast journey, and we reach our final interval of destinations. In this section, we aim to understand the underlying significances, associations, and trends that a surface-level exploration fails to capture. 
#### We go beyond our Royal Contract, and delve even deeper into the *UNKNOWN*...
* We begin by targeting our top contending classifiers to revisit our explorations, and understand the intricate relationships between all of our Scholars’ features and characteristics so that we can better improve prediction for the Kingdom!
* 1) With our Feature Importance Identification, we designed this function of deepening to extract and conceptualize the score of importance for each Scholar feature. Our plots sort the features by importance, returns this DataFrame, and extracts this importance to be mapped. We decided that identifying the Top 10 most important features was a paramount strategy to assist in the final most serious and accurate predictions. We fit the Random Forest Classifier with our top-picked hyperparameters and retrieved the feature importance scores. By forming a dataframe cataloging the feature names with their importance scores, we sorted them in descending order, with the most important ones prevailing on top. We proceed to plot them in a horizontal bar chart and can this simply and spatially understand the ranges of importance, just within this Top 10 exposition. 
* **Our Observations**: 
* `Fae Dust Reserve` is our most influential feature, by far. This is then followed by `Mystic Energy Level`, and `Mystical Index`. These are features that are closely related to a metric of magic, a trait that seems to be heavily deterministic for predicting Guild classes. One can assume that a certain amount of mystical powers/abilities are strong indicators of “legacy” for entering a certain guild. This presence leads us to assume that this mystical set of traits could transcend possible discrepancies in perfect stamina, health, or physical capabilities, as magical consistency acts potentially as a function of mystical “nepotism”, so to speak. Let’s break down these importances:

**You have entered the Black Market**
* **Welcome to our Inside Scoop: *Scholar Counseling Services***
* **CAUTION: THIS IS NOT KINGDOM-CERTIFIED; USE AT YOUR OWN RISK!**
You may… or may not have what it takes to reach your shining Master Guild goal. But if you follow our empirically-based evidence, you can “insider-trade” your way into your dreams! Yes, take *heavy* precaution, as this isn’t too legal, but can give you extreme, advanced insight into tried-and-true results. 
The most determinant features, according to a finely-tuned algorithm model (Random Forest), tells our black-market researchers that access to magical resources and power, as well as access to rare mystical elements, is highly indicative in classification motivation. 
* *From the Village Gossip*:  A little birdie told us that Scholars with higher power reserves are much more likely to be placed in higher guilds. Are you stuck without a Guild to your name? Do not fret! You can stop disappointing your family members and find a connection to increase your access to these rare resources. **Contact us if you’re willing to take the risk!**
Don’t say we didn’t warn you, some traits are just too hard to attain, even through the black market. Our researchers have informed us that Mystic Energy Levels, or your internal capacity for magic is a near direct reflection of your proficiency to perform dangerous magical tasks. Sorry not sorry if you aren’t blessed with this trait –you can try to maximize your potential with the next piece of advice.
* Mystical Indexes can be optimized with non-stop training and effort. If you practice 24/7, we have faith in you, dear Scholar! The higher your composite score of magical skill, the more inclined the Classifiers are to believe that you possess this “inner talent”.... They don’t have to know the truth though– work hard and dreams can come true!

* **The Less Important Scoop**:
* In less important, and easily-accessible, around-the-bush advice, the more wise you are, the more likely you are to be highly-ranked. This is just a fact– call it ageist; however, you older folks might love this piece of information!
* And to everyone’s knowledge, you must **always** exercise! Healthy meals and physical training end up having some kind of impact on your Guild Membership. 
* Though you, dear Reader, are on the black market searching for help, I will inform you that Knightly Valor is a great attribute for classification. If you represent courage, integrity, and leadership, you increase your chances to enter the **Master Guild**! *What are you doing here, anyway. Stop breaking the rules! Maybe that’s why you haven’t entered the top Guild yet!*

**You have exited the Black Market**

* 2) Top Features and their Boxplots, associated by Guild Membership. We felt compelled to justify our actions, as to be completely in line with our Royal Contract. In an effort to further support our selection of the Random Forest Classifier as our best performing model, we decided to visually conceptualize the distribution of the top 3 important features, sorted by class. These box plots provide a clear understanding of how these features impact the Guild Membership. 
* Our observations:
* It must be noted that in this kind of exploration, there is a slight competition of the Mystical Index amongst the others, in influence for the Guild classifications. 
* It must also be noted that the overlap between the Fae Dust Reserve and the Mystic Energy Levels explain that there are very subtle interactions among features, in terms of predicting classes. This is **precisely** WHY we chose the Random Forest model for prediction! Its complex nature and ability to sift through intricate, non-linear relationships are direct reasons to justify our final decision with the `Random Forest`!
* 3) Evaluating our previous Top 2 Contenders to continue justifying, in empirical comparisons, as to why Random Forest is the best choice in every case! We conducted these comparisons based on:
* `Mean Squared Error (MSE)`: to see how similar the predictions were to the actual values, with a lower value being a closer result!
* `R^2 Score`: showing us how successful the model is at defining variance in the Guild Membership (target variable), with higher values being the most accurate.
* `Runtime`: computational time
* Our observations: 
**MSE Values**: 
- Random Forest: low MSE in comparison
- Logistic Regression: higher than Random Forest, proving it does not predict as well
**R^2 Scores**: 
- Random Forest: achieves a high score, proving that it can define much more variance within our complex dataset, and is able to identify relationships in our target variable.
- Logistic Regression: lower score, indicating that this model is likely underfitting and not adept at defining variance.
* These metrics are obviously significant to further clarify our assurance and confidence in selecting Random Forest as our **crowned** model! 
* Though Random Forest takes slightly more time to compute, we must prioritize all of these accuracies and performance statistics over simple rate of speed. This demonstrates our understanding and grounded decision-making. 

* 4) We wanted to capture global data variance in a lower-dimension space, and visualize linear transformations from our covariance matrix. We were able to accomplish this with Principal Component Analysis (PCA). We project our X features into 2D with Principal Component 1 and 2. We then output  a scatter plot to highlight how the data points shift away from each other by the Membership classification within this reduced space. We print our important features and delineate them from the total number of features. By visualizing this Guild separation, we can understand how well it was completed.
* Our observations: 
- We notice significant overlap between the 3 Guilds in our transformed space. From this we can gather that the memberships are not entirely linearly separable based on the Features the Kingdom supplied us with. 
- This tells us that Apprentices, and even non-ranked Scholars have a somewhat chance to reach a Master Guild status, leaving them with more hope! However, this also indicates that Master Scholars have extra special or enhanced qualities within these features to have accessed their status. 
- **Though they overlap**, we can determine that some trends in this plot prove that Scholars closer to a chance to be in the Master Guild (green cluster) in this PCA display have decently “aligned” profiles with those already in the Master Guild, insinuating that they have potential to enter the Master Guild soon. 

* **ATTENTION ASPIRING SCHOLARS**:
* You must enhance your key/critical powers and strengths that are similar to your Master Guild colleagues and counterparts.
* Analyze the routines and trends that your Master Scholar friends follow, and you could improve your chance to be one of them soon!



## **<h2 align="center"> ★ THE TASK IS DONE 🔮 ★ </h2>** 

In the mystical realm of **Marendor**, where the winds whisper ancient secrets and the stars chart the destiny of wizards, a monumental task was bestowed upon us: to uncover the chosen few worthy of ascending to the illustrious **Master Guild** of Magic. Through the merging of wisdom, innovation, and tireless determination, we have brought forth a masterpiece of predictive science—a system that unravels the mysteries of guild membership with unparalleled precision.
Our journey began with the raw, unrefined tales of thousands of scholars, each carrying their hopes, dreams, and magical potential within the folds of data. We embarked on a meticulous quest to transform this chaos into clarity, wielding the finest tools of data science like enchanted relics of old. Through the alchemy of correlation analysis, chi-squared tests, and outlier detection, we polished the hidden gems of information, shaping a dataset worthy of royal scrutiny.
At the heart of our endeavor stood the **Random Forest**, the crown jewel of machine learning models. Like a wise and just arbiter, it emerged victorious from trials against formidable adversaries—**Gradient Boosted Trees, Logistic Regression, and K-Nearest Neighbors**. Through rigorous training, validation, and the arcane art of hyperparameter tuning, the **Random Forest** proved its mettle, illuminating the path to the Master Guild with unwavering accuracy and adaptability.
As we delved deeper into the mystical web of scholar attributes, certain traits revealed their true power: the ethereal **Fae Dust Reserve**, the boundless **Mystic Energy Level**, and the enigmatic **Mystical Index**. These features, akin to the ancient runes etched into the stones of Marendor, held the secrets to guild membership, transcending the mundane and guiding us toward the truly gifted.
But our work did not stop at mere prediction. With the precision of a master cartographer, we charted the intricate relationships between a scholar’s traits and their magical destiny. Each insight became a beacon, shedding light on the qualities that distinguish the worthy from the rest and offering a roadmap for the kingdom's future.
This is not just a dataset system—it is a legacy. A dynamic, ever-adaptive framework that ensures the Master Guild of Magic will be graced by the most deserving wizards, while fostering a culture of excellence and ambition within the magical community. With this system in place, the kingdom of Marendor now stands at the precipice of a golden age, where data and magic unite to secure its future.
Your Majesty, we have not only fulfilled the royal decree but have elevated it to a masterpiece of innovation and foresight. The Master Guild will no longer be shrouded in mystery; its gates will open only to those truly destined to lead. And as the sun sets on this chapter, the kingdom shines brighter than ever, its future etched in the stars and guided by the wisdom of data.
### **Long live the King, and may the magic of Marendor prosper for eternity!**




