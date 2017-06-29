
import java.io.IOException;
import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.api.java.UDF2;
import org.apache.spark.sql.catalyst.expressions.ConcatWs;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.dmg.pmml.DataType;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.sql.functions;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
public class gbTree2 {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
		// For Location
		Dataset<Row> training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allData.csv");
		Dataset<Row> data = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/data.csv");
		Dataset<Row> ld1 = data.select(data.col("locality_detail1").as("loc")).distinct();
		Dataset<Row> ld2 = data.select(data.col("locality_detail2").as("loc")).distinct();
		Dataset<Row> ld3 = data.select(data.col("locality_detail3").as("loc")).distinct();

//		System.out.println(training.count());
		// For Location
		
		Dataset<Row> city = training.select(training.col("redemptionAddress_city_name").as("loc")).distinct();
		Dataset<Row> location = training.select(training.col("redemptionAddress_locality_name").as("loc")).distinct();
		System.out.println(city.count() + " " + location.count());
		Dataset<Row> Location_training1 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/location.csv");
		Dataset<Row> Location_training2 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allLocality.csv");
		Dataset<Row> Location_training3 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allMalls.csv");
		Location_training1 = Location_training1.select(Location_training1.col("redemptionAddress_locality_name").as("loc"));
		Location_training2 = Location_training2.select(Location_training2.col("Locations").as("loc"));
		Location_training3 = Location_training3.select(Location_training3.col("Localities").as("loc"));
		
		Dataset<Row> Location_training = location.union(city).union(Location_training1).union(Location_training2).union(Location_training3).union(ld1).union(ld2).union(ld3).distinct();
		System.out.println(Location_training.count());
		Location_training = Location_training.filter(Location_training.col("loc").isNotNull());
//		Location_training = Location_training.union(Location_training3).un.distinct();
		Tokenizer Location_tokenizer=new Tokenizer().setInputCol("loc").setOutputCol("words");				  
		Dataset<Row> Location_output=Location_tokenizer.transform(Location_training);
		StopWordsRemover Location_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Location_afterStop = Location_remover.transform(Location_output);
		int numFeatures = 10;
		HashingTF Location_hashingTF = new HashingTF()
				  .setInputCol("words2")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> Location_featurizedData = Location_hashingTF.transform(Location_afterStop);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

		IDF Location_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel Location_idfModel = Location_idf.fit(Location_featurizedData);

		Dataset<Row> Location_rescaledData = Location_idfModel.transform(Location_featurizedData);
		Dataset<Row> Location_vector = Location_rescaledData;
		
		

		/*NGram Location_unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> Location_UniGrams=Location_unigram.transform(Location_afterStop);
		NGram Location_bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> Location_BiGrams = Location_bigram.transform(Location_afterStop);
		Dataset<Row> Location_uni = Location_UniGrams.select("Unigrams");
		Dataset<Row> Location_bi = Location_BiGrams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> Location_finalGrams= Location_uni.union(Location_bi).distinct();
		Word2Vec Location_word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(5).setMinCount(0);
		Word2VecModel Location_model = Location_word2Vec.fit(Location_finalGrams);
		Dataset<Row> Location_vector = Location_model.transform(Location_finalGrams).select("Unigrams","Vectors");  */
		Location_vector = Location_vector.withColumn("Label", functions.lit(0.0));		
		Location_vector = Location_vector.select(Location_vector.col("words2").as("Words"),Location_vector.col("rawFeatures"),Location_vector.col("features"),Location_vector.col("Label").as("label"));

//		System.out.println(Location_vector.count());

			
		//		Location_vector.show(false);

		/*double[] array = {0.7,0.2,0.1};
		Dataset<Row>[] Location_splitting = Location_vector.randomSplit(array, 5065);
		Dataset<Row> Location_Training = Location_splitting[0];
		Dataset<Row> Location_Testing  = Location_splitting[1];
		Dataset<Row> Location_Validation = Location_splitting[2];
		*/
		//For Merchant
		
		Dataset<Row> Merchant_training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/merchants.csv").distinct();
		Dataset<Row> temp = training.select("name");
		Merchant_training = Merchant_training.union(temp).distinct();
//		System.out.println(training.count());
		Tokenizer Merchant_tokenizer=new Tokenizer().setInputCol("name").setOutputCol("words");				  
		Dataset<Row> Merchant_output=Merchant_tokenizer.transform(Merchant_training);
		StopWordsRemover Merchant_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Merchant_afterStop = Merchant_remover.transform(Merchant_output);
		
		HashingTF Merchant_hashingTF = new HashingTF()
				  .setInputCol("words2")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> Merchant_featurizedData = Merchant_hashingTF.transform(Merchant_afterStop);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

		IDF Merchant_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
		IDFModel Merchant_idfModel = Merchant_idf.fit(Merchant_featurizedData);

		Dataset<Row> Merchant_rescaledData = Merchant_idfModel.transform(Merchant_featurizedData);
		Dataset<Row> Merchant_vector = Merchant_rescaledData;
		/*NGram Merchant_unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> Merchant_UniGrams=Merchant_unigram.transform(Merchant_afterStop);
		NGram Merchant_bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> Merchant_BiGrams = Merchant_bigram.transform(Merchant_afterStop);
		Dataset<Row> Merchant_uni = Merchant_UniGrams.select("Unigrams");
		Dataset<Row> Merchant_bi = Merchant_BiGrams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> Merchant_finalGrams= Merchant_uni.union(Merchant_bi).distinct();
		Word2Vec Merchant_word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(5).setMinCount(0);
		Word2VecModel Merchant_model = Merchant_word2Vec.fit(Merchant_finalGrams);
		Dataset<Row> Merchant_vector = Merchant_model.transform(Merchant_finalGrams).select("Unigrams","Vectors"); */ 
		Merchant_vector = Merchant_vector.withColumn("Label", functions.lit(1.0));		
		Merchant_vector = Merchant_vector.select(Merchant_vector.col("words2").as("Words"),Merchant_vector.col("rawFeatures"),Merchant_vector.col("features"),Merchant_vector.col("Label").as("label"));
		
		/*Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(array, 5065);
		Dataset<Row> Merchant_Training = Merchant_splitting[0];
		Dataset<Row> Merchant_Testing  = Merchant_splitting[1];
		Dataset<Row> Merchant_Validation = Merchant_splitting[2];
		*/
//		double[] arr = {.12,.88};
//		Dataset<Row>[] splitting = Location_vector.randomSplit(arr, 10000L);
//		Location_vector = splitting[0];
		double[] fifty = {0.8,0.2};
		Dataset<Row>[] Location_splitting = Location_vector.randomSplit(fifty);
		Dataset<Row> Location_trainingDF = Location_splitting[0];
		Dataset<Row> Location_testingDF  = Location_splitting[1];
		
		Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(fifty);
		Dataset<Row> Merchant_trainingDF = Merchant_splitting[0];
		Dataset<Row> Merchant_testingDF  = Merchant_splitting[1];

		Dataset<Row> trainingDF = Location_trainingDF.union(Merchant_trainingDF);
		Dataset<Row> testingDF  = Location_testingDF.union(Merchant_testingDF);
//		Dataset<Row> validationDF = Location_validationDF.union(Merchant_validationDF);
		GBTClassifier gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setMaxIter(10); 
		GBTClassificationModel gbtModel = gbt.fit(trainingDF);
//		RandomForestClassifier forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features");
//		RandomForestClassificationModel forestModel  = forest.fit(trainingDF);
		
		Dataset<Row> predictions = gbtModel.transform(testingDF);
//		predictions.show(false);
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				  .setLabelCol("label")
				  .setPredictionCol("prediction")
				  .setMetricName("accuracy");
		double accuracy = evaluator.evaluate(predictions);
		
		
		
		System.out.println("Test set accuracy = " + accuracy + " count = " + predictions.filter(predictions.col("prediction").equalTo(0.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(0.0)).count() + " count = " + predictions.filter(predictions.col("prediction").equalTo(1.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(1.0)).count());

		try {
			gbtModel.write().overwrite().save("/Users/nishantarora/Downloads/gbTree2");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}


		
	

}
