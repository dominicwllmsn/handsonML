Machine Learning: 
	- the field of study which gives computers the ability to learn without being explicitly programmed.
	- a program is said to learn from experience E (training data) w.r.t. a task T, and some perfomance
	measure P, if its performance on T, measured by P, improves with experience E.

An ML system is useful as it can automatically learn relevant features (such as words or phrases which are good
predictors of spam in emails) whereas an explicit program would require a lot of finetuning/complex rules. Problems
which are too complex for analytical approaches or for which there is no known algorithm/pattern are also suitable for ML.
Further, an ML system can adapt to new data, meaning that fluctuating environments (like the identifiers of spam emails) can be
accounted for in the program. Some ML systems can also be analysed to see what they have learnt, potentially revealing 
unsuspected correlations or trends, leading to a better understanding of the problem.

NLP examples: Chatbots, identifying offensive comments
CV examples: Classification of products on production line, detecting tumours (semantic segmentation)
Regression: Forecasting revenue based on performance metrics
Clustering: Segmenting clients for targeted marketing based on purchases


Types of ML systems:

	Supervised: the training set fed to the algorithm includes the desired solutions (labels).
		Typically used in classification. Also regression to predict a numerical value based on a set of features.
		Models include k-NN, linear & logistic regression, SVMs, Decision Trees & Random Forests, neural networks

	Unsupervised: the training data is unlabeled.
		Models include clustering (K-means etc.), anomaly detection (one-class SVM), visualisation & dim. reduction
		(PCA etc.) and association rule learning. Dimensioanlity reduction simplifies data without losing too much info
		such as by merging several correlated features into one. Association rule learning discovers interesting relations
		between attributes.

	Semisupervised: partially labelled data - typically a large amount of unlabeled data and a few labeled instances.
		Often combinations of unsup & sup algorithms, e.g. DBNs are based on RBMs stacked on eachother. They are 
		trained sequentially unsupervised, then the whole system is finetuned using supervised learning techniques.

	Reinforcement: the learning system (agent) can observe the environment, select and perform actions, and get rewards/penalties
		in return. It then learns the best strategy (policy) to get the most reward over time. A policy defines what action the agent
		chooses in a given situation.

	Batch: the system is incapable of learning incrementally - it must be trained on all available data (i.e. offline). It must
		be trained from scratch if you need it to know about new data.

	Online: the system is trained incrementally by feeding training instances sequentially, either individually or in mini-batches. Good
		for systems with near-continuous data flow which need to change rapidly/autonomously. Also useful with limited computational 
		resource or when there is so much data it cannot fit in one machine's main memory (out-of-core learning) - so online = incremental.
		Involves a learning rate dictating how fast it adapts to new data - high means it quickly forgets old data, low means there is 
		an inertia to learning, but less sensitive to noise or outliers in the new data.

	Instance-based: the system learns the examples by heart, then generalises to new cases by using a similarity measure to compare
		them to learned examples (or a subset of them) - e.g. k-NN or radial basis functions. 

	Model-based: the system generalises to new data by building a model of the training examples, then uses to model to make predictions.
		This is acheived by the algorithm minimising a cost function - a performance measure which measures how bad the current model 
		performs on the training data. It is also possible to define a fitness/utility function to measure how good the model is.


Challenges of ML systems:
 DATA	
	Insufficient quantity of data: ML requires a lot of data for the algorithms to generalise well and avoid fitting noise. 
		See: The Unreasonable Effectiveness of Data by Norvig et al.

	Nonrepresentative training data: to generalise well, the training data must be representative of the new cases we want to generalise
		to - true for instance and model-based learning. Using non-representative data means the trained model is unlikely to make
		accurate predictions out of sample. N.B. if the sample is too small, we have sampling noise (nonrepresentative data as a result
		of chance), while even large datasets can be nonrepresentative if the sampling method is flawed (sampling bias).

	Poor-quality data: if the dataset if full of errors, outliers and noise (due to poor quality measurements) it will be harder for the 
		model to detect the underlying patter, so less likely to generalise well. So, cleaning the data is important, e.g. (1) if some
		instances are clear outliers, discard or fix them manually, (2) if some instances are missing features, decide whether to ignore
		the attribute altogether, fill in the missing values (e.g. with a median), or train one model with the feature and one without.

	Irrelevant features: the system is only capable of learning if the training data containis enough relevant features and not too many
		irrelevant ones - need to use feature engineering to select a good set of features to train on, via (1) feature selection (select
		the most useful features from the existing), (2) feature extraction (combining existing features, like in dim reduction), or (3)
		by creating new features from new data. 

 ALGORITHM		
	Overfitting: the model performs well in sample but does not generalise well. Complex models such as deep NN can detect subtle patterns,
		but if the training data is noisy/too small (so sampling noise), the model is likely to fit the noise, so generalises poorly.
		This also occurs when training using irrelevant features by fitting non-existent patterns. Solutions are (1) simplifying the 
		model (fewer parameters)/reducing the number of attributes in the data, or constraining the model (regularization), (2) 
		more training data, or (3) reducing the noise the training data (e.g. fix data errors, remove outliers).

	Underfitting: the model is too simple to learn the underlying structure of the data. Solutions are (1) selecting a more powerful model
		(more parameters), (2) feed better features to the algorithm (feature engineering), or (3) reducing the constraints on the model
		(e.g. reduce the regularization parameter).


Testing and Validating:
	To test how a model generalises, split the data into a training and test set and find the out-of-sample error by evaluating the final 
	trained model on the test set. Generally there is an 80/20 split between training and test data, depending on the size of the dataset 
	(more data means a larger proportion can be assigned for training).

	Hyperparameter tuning and model selection: Validation is used to select the best hyperparameters for the model without having to touch
		the test set (which would jeopardise the generalization in production) - evaluating on the test set must be left to the very end,
		when all the hyperparameters are selected. Validation is usually done via cross-validation: each model is trained on a reduced 
		training set (minus a small validation set, which it is then evaluated on). By averaging out the evaluations for a model, you get
		a more accurate measure of performance. However, the training time is multiplied by the number of validation sets.

	Data mismatch: In some cases, it’s easy to get a large amount of data for training, but this data probably won’t be perfectly 
		representative of the data that will be used in production. For example, suppose you want to create a mobile app to take 
		pictures of flowers and automatically determine their species. You can easily download millions of pictures of flowers on 
		the web, but they won’t be perfectly representative of the pictures that will actually be taken using the app on a mobile device.
		Perhaps you only have 10,000 representative pictures (i.e., actually taken with the app). In this case, the most important rule 
		to remember is that the validation set and the test set must be as representative as possible of the data you expect to use in 
		production, so they should be composed exclusively of representative pictures: you can shuffle them and put half in the 
		validation set and half in the test set (making sure that no duplicates or near-duplicates end up in both sets). But after 
		training your model on the web pictures, if you observe that the performance of the model on the validation set is 
		disappointing, you will not know whether this is because your model has overfit the training set, or whether this is just 
		due to the mismatch between the web pictures and the mobile app pictures. One solution is to hold out some of the 
		training pictures (from the web) in yet another set that Andrew Ng calls the train-dev set. After the model is trained 
		(on the training set, not on the train-dev set), you can evaluate it on the train-dev set. If it performs well, then the 
		model is not overfitting the training set. If it performs poorly on the validation set, the problem must be coming from the 
		data mismatch. You can try to tackle this problem by preprocessing the web images to make them look more like the pictures that 
		will be taken by the mobile app, and then retraining the model. Conversely, if the model performs poorly on the train-dev set, 
		then it must have overfit the training set, so you should try to simplify or regularize the model, get more training data, and 
		clean up the training data.

	No Free Lunch Theorem: A model is a simplified version of the observations. The simplifications are meant to discard the superfluous 
		details that are unlikely to generalize to new instances. To decide what data to discard and what data to keep, you must make
		assumptions. For example, a linear model makes the assumption that the data is fundamentally linear and that the distance between
		the instances and the straight line is just noise, which can safely be ignored. In a famous 1996 paper, David Wolpert 
		demonstrated that if you make absolutely no assumption about the data, then there is no reason to prefer one model over 
		any other. This is called the No Free Lunch (NFL) theorem. For some datasets the best model is a linear model, while for 
		other datasets it is a neural network. There is no model that is a priori guaranteed to work better (hence the name of the 
		theorem). The only way to know for sure which model is best is to evaluate them all. Since this is not possible, in practice 
		you make some reasonable assumptions about the data and evaluate only a few reasonable models. For example, for simple tasks 
		you may evaluate linear models with various levels of regularization, and for a complex problem you may evaluate various 
		neural networks.

