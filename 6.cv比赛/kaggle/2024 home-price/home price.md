#home price

##格式，数据内容

提交文件格式，该文件应包含标题，并采用以下格式：

	Id,SalePrice
	1461,169000.1
	1462,187724.1233
	1463,175221
	etc.

文件描述

	train.csv - 训练集
	test.csv - 测试集
	data_description.txt - 每列的完整描述，最初由 Dean De Cock 准备，但经过轻微编辑以匹配此处使用的列名称
	sample_submission.csv - 根据销售年份和月份、地块面积和卧室数量的线性回归提交的基准

数据字段

	SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
	MSSubClass: The building class
	MSZoning: The general zoning classification
	LotFrontage: Linear feet of street connected to property
	LotArea: Lot size in square feet
	Street: Type of road access
	Alley: Type of alley access
	LotShape: General shape of property
	LandContour: Flatness of the property
	Utilities: Type of utilities available
	LotConfig: Lot configuration
	LandSlope: Slope of property
	Neighborhood: Physical locations within Ames city limits
	Condition1: Proximity to main road or railroad
	Condition2: Proximity to main road or railroad (if a second is present)
	BldgType: Type of dwelling
	HouseStyle: Style of dwelling
	OverallQual: Overall material and finish quality
	OverallCond: Overall condition rating
	YearBuilt: Original construction date
	YearRemodAdd: Remodel date
	RoofStyle: Type of roof
	RoofMatl: Roof material
	Exterior1st: Exterior covering on house
	Exterior2nd: Exterior covering on house (if more than one material)
	MasVnrType: Masonry veneer type
	MasVnrArea: Masonry veneer area in square feet
	ExterQual: Exterior material quality
	ExterCond: Present condition of the material on the exterior
	Foundation: Type of foundation
	BsmtQual: Height of the basement
	BsmtCond: General condition of the basement
	BsmtExposure: Walkout or garden level basement walls
	BsmtFinType1: Quality of basement finished area
	BsmtFinSF1: Type 1 finished square feet
	BsmtFinType2: Quality of second finished area (if present)
	BsmtFinSF2: Type 2 finished square feet
	BsmtUnfSF: Unfinished square feet of basement area
	TotalBsmtSF: Total square feet of basement area
	Heating: Type of heating
	HeatingQC: Heating quality and condition
	CentralAir: Central air conditioning
	Electrical: Electrical system
	1stFlrSF: First Floor square feet
	2ndFlrSF: Second floor square feet
	LowQualFinSF: Low quality finished square feet (all floors)
	GrLivArea: Above grade (ground) living area square feet
	BsmtFullBath: Basement full bathrooms
	BsmtHalfBath: Basement half bathrooms
	FullBath: Full bathrooms above grade
	HalfBath: Half baths above grade
	Bedroom: Number of bedrooms above basement level
	Kitchen: Number of kitchens
	KitchenQual: Kitchen quality
	TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
	Functional: Home functionality rating
	Fireplaces: Number of fireplaces
	FireplaceQu: Fireplace quality
	GarageType: Garage location
	GarageYrBlt: Year garage was built
	GarageFinish: Interior finish of the garage
	GarageCars: Size of garage in car capacity
	GarageArea: Size of garage in square feet
	GarageQual: Garage quality
	GarageCond: Garage condition
	PavedDrive: Paved driveway
	WoodDeckSF: Wood deck area in square feet
	OpenPorchSF: Open porch area in square feet
	EnclosedPorch: Enclosed porch area in square feet
	3SsnPorch: Three season porch area in square feet
	ScreenPorch: Screen porch area in square feet
	PoolArea: Pool area in square feet
	PoolQC: Pool quality
	Fence: Fence quality
	MiscFeature: Miscellaneous feature not covered in other categories
	MiscVal: $Value of miscellaneous feature
	MoSold: Month Sold
	YrSold: Year Sold
	SaleType: Type of sale
	SaleCondition: Condition of sale

##设计思路：

对于连续的数据，比如价格可以先求出均值和方差等等特征。然后normalization做标准化（standardization）：设该特征在整个数据集上的均值为μ，标准差为σ。那么，我们可以将该特征的每个值先减去μ再除以σ得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。

标准化后，每个数值特征的均值变为0，所以可以直接用0代替缺失值,即缺失值NA和无意义值NAN用0代替

然后处理离散的数据。假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。

将 分类变量 转换为 独热编码（One-Hot Encoding）

使用
	
	get_dummies即上述将离散值转换为指示特征
	pd.get_dummies
	

然后转化张量特征

	# 训练集特征
	train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
	# 测试集特征
	test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
	# 训练集标签
	train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

##方法设计

用简单的回归问题

初始化net网络

	def get_net(feature_num):
		# 实例化nn
	    net = nn.Linear(feature_num, 1)
	    for param in net.parameters():
	        nn.init.normal_(param, mean=0, std=0.01)
	    return net

均方差损失

	def log_rmse(net, features, labels):
	    with torch.no_grad():
	        # 将小于1的值设成1，使得取对数时数值更稳定
	        clipped_preds = torch.max(net(features), torch.tensor(1.0))
	        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
	    return rmse.item()

