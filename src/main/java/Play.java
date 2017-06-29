
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

public class Play {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
		// For Location
		
		Dataset<Row> Location_training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/location.csv").distinct();
//		Dataset<Row> Location_training2 = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/location.csv").distinct();

//		System.out.println(training.count());
		Tokenizer Location_tokenizer=new Tokenizer().setInputCol("redemptionAddress_locality_name").setOutputCol("words");				  
		Dataset<Row> Location_output=Location_tokenizer.transform(Location_training);
		StopWordsRemover Location_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Location_afterStop = Location_remover.transform(Location_output);
		NGram Location_unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> Location_UniGrams=Location_unigram.transform(Location_afterStop);
		NGram Location_bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> Location_BiGrams = Location_bigram.transform(Location_afterStop);
		Dataset<Row> Location_uni = Location_UniGrams.select("Unigrams");
		Dataset<Row> Location_bi = Location_BiGrams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> Location_finalGrams= Location_uni.union(Location_bi).distinct();
		Word2Vec Location_word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(5).setMinCount(0);
		Word2VecModel Location_model = Location_word2Vec.fit(Location_finalGrams);
		Dataset<Row> Location_vector = Location_model.transform(Location_finalGrams).select("Unigrams","Vectors");  
		Location_vector = Location_vector.withColumn("Label", functions.lit(0.0));		
		Location_vector = Location_vector.select(Location_vector.col("Unigrams").as("Grams"),Location_vector.col("Vectors").as("features"),Location_vector.col("Label").as("label"));
		/*double[] array = {0.7,0.2,0.1};
		Dataset<Row>[] Location_splitting = Location_vector.randomSplit(array, 5065);
		Dataset<Row> Location_Training = Location_splitting[0];
		Dataset<Row> Location_Testing  = Location_splitting[1];
		Dataset<Row> Location_Validation = Location_splitting[2];
		*/
		//For Merchant
		
		Dataset<Row> Merchant_training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/merchants.csv").distinct();
//		System.out.println(training.count());
		Tokenizer Merchant_tokenizer=new Tokenizer().setInputCol("name").setOutputCol("words");				  
		Dataset<Row> Merchant_output=Merchant_tokenizer.transform(Merchant_training);
		StopWordsRemover Merchant_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Merchant_afterStop = Merchant_remover.transform(Merchant_output);
		NGram Merchant_unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> Merchant_UniGrams=Merchant_unigram.transform(Merchant_afterStop);
		NGram Merchant_bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> Merchant_BiGrams = Merchant_bigram.transform(Merchant_afterStop);
		Dataset<Row> Merchant_uni = Merchant_UniGrams.select("Unigrams");
		Dataset<Row> Merchant_bi = Merchant_BiGrams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> Merchant_finalGrams= Merchant_uni.union(Merchant_bi).distinct();
		Word2Vec Merchant_word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(5).setMinCount(0);
		Word2VecModel Merchant_model = Merchant_word2Vec.fit(Merchant_finalGrams);
		Dataset<Row> Merchant_vector = Merchant_model.transform(Merchant_finalGrams).select("Unigrams","Vectors");  
		Merchant_vector = Merchant_vector.withColumn("Label", functions.lit(1.0));		
		Merchant_vector = Merchant_vector.select(Merchant_vector.col("Unigrams").as("Grams"),Merchant_vector.col("Vectors").as("features"),Merchant_vector.col("Label").as("label"));
		
		/*Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(array, 5065);
		Dataset<Row> Merchant_Training = Merchant_splitting[0];
		Dataset<Row> Merchant_Testing  = Merchant_splitting[1];
		Dataset<Row> Merchant_Validation = Merchant_splitting[2];
		*/
		double[] array = {0.7,0.2,0.1};
		/*double [] m = {0.05,0.95};
		Dataset<Row>[] merchantSplit  = Merchant_vector.randomSplit(m);
		Merchant_vector = merchantSplit[0];*/
		Dataset<Row> mergedDF =  Location_vector.union(Merchant_vector);
		
		MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures");
		MinMaxScalerModel scalerModel = scaler.fit(mergedDF);
		mergedDF = scalerModel.transform(mergedDF);
		mergedDF = mergedDF.select(mergedDF.col("Grams"),mergedDF.col("scaledFeatures").as("features"),mergedDF.col("label"));
		Dataset<Row>[] merged_splitting = mergedDF.randomSplit(array);
		Dataset<Row> trainingDF = merged_splitting[0];
		Dataset<Row> testingDF  = merged_splitting[1];
		Dataset<Row> validationDF = merged_splitting[2];
		NaiveBayes nb = new NaiveBayes();		
		NaiveBayesModel model = nb.fit(trainingDF);
		Dataset<Row> predictions = model.transform(testingDF);
//		predictions.show(false);
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				  .setLabelCol("label")
				  .setPredictionCol("prediction")
				  .setMetricName("accuracy");
				double accuracy = evaluator.evaluate(predictions);
		System.out.println("Test set accuracy = " + accuracy + " count = " + predictions.filter(predictions.col("prediction").equalTo(0.0)).count());
//		predictions.filter(predictions.col("prediction").equalTo(0.0)).count();
		
		/*LogisticRegression lr = new LogisticRegression().setMaxIter(20).setRegParam(0.005);
		LogisticRegressionModel lrModel = lr.fit(mergedDF);
//		lrModel.setThreshold(0.5);
//		lr.setThreshold(0.5);
		//Training.show(false);
		Dataset<Row> predictionss = lrModel.transform(mergedDF);
		predictionss.filter(predictionss.col("prediction").equalTo(0.0)).show(false);*/
		
		
//		trainingDF.filter(trainingDF.col("label").equalTo(0.0)).show(false);

//		predictions.show(false);
//		System.out.println("Coefficients: "+ lrModel.coefficients() + " Intercept: " + lrModel.intercept());	
//		System.out.println(Location_training.count() + "" + Merchant_training.count());
//		System.out.println(mergedDF.count() + " " + trainingDF.count() + " " + testingDF.count() + " " + validationDF.count());
//		mergedDF.filter(mergedDF.col("label").equalTo(1.0)).show(250);


		
	}

}
