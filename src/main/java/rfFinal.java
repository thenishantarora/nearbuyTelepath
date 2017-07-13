import java.io.IOException;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class rfFinal {

	
	
	public static void main(String[] args) {
		
		// TODO Auto-generated method stub
		
	// Loading the Spark Configuration	
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local[4]");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
	// Loading the location databases and merging them			
		Dataset<Row> training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allData.csv");		
		Dataset<Row> NCR = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/NCR.csv");
		Dataset<Row> city = training.select(training.col("redemptionAddress_city_name").as("loc")).distinct();
		Dataset<Row> location = training.select(training.col("redemptionAddress_locality_name").as("loc")).distinct();
		System.out.println(city.count() + " " + location.count());
		Dataset<Row> Location_training1 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/location.csv");
		Dataset<Row> Location_training2 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allLocality.csv");
		Dataset<Row> Location_training3 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allMalls.csv");
		Location_training1 = Location_training1.select(Location_training1.col("redemptionAddress_locality_name").as("loc"));
		Location_training2 = Location_training2.select(Location_training2.col("Locations").as("loc"));
		Location_training3 = Location_training3.select(Location_training3.col("Localities").as("loc"));
		Dataset<Row> Location_training = location.union(city).union(Location_training1).union(Location_training2).union(Location_training3).union(NCR).distinct();

	// Replicating the Location Data 4 times	
		Dataset<Row> Location_1 = Location_training;
		Dataset<Row> Location_2 = Location_training;
		Dataset<Row> Location_4 = Location_training;
		Location_training = Location_training.union(Location_1).union(Location_2).union(Location_4);
		Location_training = Location_training.filter(Location_training.col("loc").isNotNull());
		
	// Tokenizing and removing the Stop Words from the data	
		Tokenizer Location_tokenizer=new Tokenizer().setInputCol("loc").setOutputCol("words");				  
		Dataset<Row> Location_output=Location_tokenizer.transform(Location_training);
		StopWordsRemover Location_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Location_afterStop = Location_remover.transform(Location_output);
		Dataset<Row> Location_vector = Location_afterStop;

	// Adding the label 0.0 to the location data	
		Location_vector = Location_vector.withColumn("Label", functions.lit(0.0));		
		Location_vector = Location_vector.select(Location_vector.col("words2").as("Words"),Location_vector.col("Label").as("label"));
		

		
	// Loading the merchant data	
		Dataset<Row> Merchant_training = training.select("name").distinct();

	// Tokenizing and removing the Stop Words	
		Tokenizer Merchant_tokenizer=new Tokenizer().setInputCol("name").setOutputCol("words");				  
		Dataset<Row> Merchant_output=Merchant_tokenizer.transform(Merchant_training);
		StopWordsRemover Merchant_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Merchant_afterStop = Merchant_remover.transform(Merchant_output);
		Dataset<Row> Merchant_vector = Merchant_afterStop;
		
	// Adding the label 0.0 to the Merchant data	
		Merchant_vector = Merchant_vector.withColumn("Label", functions.lit(1.0));		
		Merchant_vector = Merchant_vector.select(Merchant_vector.col("words2").as("Words"),Merchant_vector.col("Label").as("label"));
		
	// Combining the location and the merchant dataframes	
		Dataset<Row> Combined_vector = Location_vector.union(Merchant_vector);
		
	// Making Unigrams and Bigrams from the combined data	
		NGram unigram = new NGram().setN(1).setInputCol("Words").setOutputCol("grams");
		NGram bigram = new NGram().setN(2).setInputCol("Words").setOutputCol("grams");
		Dataset<Row> Combined_vector1 = unigram.transform(Combined_vector);
		Dataset<Row> Combined_vector2 = bigram.transform(Combined_vector);
		Combined_vector = Combined_vector1.union(Combined_vector2);
		Combined_vector = Combined_vector.select(Combined_vector.col("grams").as("Words"),Combined_vector.col("label"));
		
	// Using Word2Vec to make features from the data	
		Word2Vec word2Vec = new Word2Vec().setInputCol("Words").setOutputCol("w2v").setVectorSize(10).setMinCount(0).setWindowSize(1);
		Word2VecModel model = word2Vec.fit(Combined_vector);
		Dataset<Row> w2v = model.transform(Combined_vector).select("Words","w2v","label");
		
	// Saving the Word2Vec Model	
		try {
			model.write().overwrite().save("/Users/nishantarora/Downloads/rfW2VModel");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		
	// Using TF-IDF to make features from the data	
		int numFeatures = 20;
		HashingTF hashingTF = new HashingTF()
				  .setInputCol("Words")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> featurizedData = hashingTF.transform(w2v);

		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
		IDFModel idfModel = idf.fit(featurizedData);
		Dataset<Row> rescaledData = idfModel.transform(featurizedData);
		Combined_vector = rescaledData;
		
		
	// Using Vector Assembler to combine the two feature vectors formed by TF-IDF and Word2Vec into a single feature vector	
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(new String[]{"w2v", "features1"})
				  .setOutputCol("features");

		Combined_vector = assembler.transform(Combined_vector);
		Combined_vector = Combined_vector.select("Words","features","label");
		
	// Saving the TF-IDF Model	
		try {
			hashingTF.write().overwrite().save("/Users/nishantarora/Downloads/RFhashModel");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		try {
			idfModel.write().overwrite().save("/Users/nishantarora/Downloads/RFidfModel");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		w2v = Combined_vector;
		
	// Splitting the raw data to training and testing data in a ratio of 9:1	
		double[] fifty = {0.9,0.1};
		Dataset<Row>[] splitting = w2v.randomSplit(fifty);
		Dataset<Row> trainingDF = splitting[0];
		Dataset<Row> testingDF  = splitting[1];

		
	// Fitting the training data into Random Forest Classification Model	
		RandomForestClassifier forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setMaxDepth(10).setNumTrees(10);
		RandomForestClassificationModel gbtModel  = forest.fit(trainingDF);
		
	// Making predictions on the testing data using the trained random forest model	
		Dataset<Row> predictions = gbtModel.transform(testingDF);
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				  .setLabelCol("label")
				  .setPredictionCol("prediction")
				  .setMetricName("accuracy");
		double accuracy = evaluator.evaluate(predictions);
		
		
	// Printing the model accuracy on the testing data	
		System.out.println("Test set accuracy = " + accuracy + " count = " + predictions.filter(predictions.col("prediction").equalTo(0.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(0.0)).count() + " count = " + predictions.filter(predictions.col("prediction").equalTo(1.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(1.0)).count());

		
	// Saving the Random Forest Classification Model	
		try {
			gbtModel.write().overwrite().save("/Users/nishantarora/Downloads/rfFinal");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}


		
	

}
