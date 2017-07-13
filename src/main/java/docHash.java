import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;

import java.io.IOException;
import java.lang.String;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
public class docHash {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
			SparkConf conf = new SparkConf().setAppName("Name").setMaster("local[4]");
			JavaSparkContext sc = new JavaSparkContext(conf);
			SparkSession ssc = new SparkSession(sc.sc());
			SQLContext spark = ssc.sqlContext();

			Dataset<Row> locations = spark.read().text("/Users/nishantarora/Desktop/locations1.txt");
			Dataset<Row> merchants = spark.read().text("/Users/nishantarora/Desktop/merchants1.txt");
			
//			locations.show(false);

//			string.show(false);
			Tokenizer Location_tokenizer = new Tokenizer().setInputCol("value").setOutputCol("tokens");
			Dataset<Row> Location_tokens = Location_tokenizer.transform(locations);
//			tokens.show(false);
//			tokens.printSchema();
			StopWordsRemover Location_swr = new StopWordsRemover().setInputCol("tokens").setOutputCol("Words");
			Dataset<Row> Location_words = Location_swr.transform(Location_tokens);
//			words.show(false);
			int numFeatures = 100;
			
			
//			Location_words.show(false);
			HashingTF hashingTF = new HashingTF()
					  .setInputCol("Words")
					  .setOutputCol("rawFeatures")
					  .setNumFeatures(numFeatures);

			Dataset<Row> Location_featurizedData = hashingTF.transform(Location_words);
					// alternatively, CountVectorizer can also be used to get term frequency vectors

			IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
			IDFModel idfModel = idf.fit(Location_featurizedData);

			Dataset<Row> Location_rescaledData = idfModel.transform(Location_featurizedData);
			Location_words = Location_rescaledData;
			
			Location_words = Location_words.withColumn("Label", functions.lit(0.0));		
			Location_words = Location_words.select(Location_words.col("Words"),Location_words.col("features"),Location_words.col("Label").as("label"));
			
			Tokenizer Merchant_tokenizer = new Tokenizer().setInputCol("value").setOutputCol("tokens");
			Dataset<Row> Merchant_tokens = Merchant_tokenizer.transform(merchants);
//			tokens.show(false);
//			tokens.printSchema();
			StopWordsRemover Merchant_swr = new StopWordsRemover().setInputCol("tokens").setOutputCol("Words");
			Dataset<Row> Merchant_words = Merchant_swr.transform(Merchant_tokens);
//			words.show(false);
//			int numFeatures = 100;
			
			
			

			Dataset<Row> Merchant_featurizedData = hashingTF.transform(Merchant_words);
					// alternatively, CountVectorizer can also be used to get term frequency vectors

//			IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//			IDFModel idfModel = idf.fit(Merchant_featurizedData);

			Dataset<Row> Merchant_rescaledData = idfModel.transform(Merchant_featurizedData);
			Merchant_words = Merchant_rescaledData;
			Merchant_words = Merchant_words.withColumn("Label", functions.lit(1.0));		
			Merchant_words = Merchant_words.select(Merchant_words.col("Words"),Merchant_words.col("features"),Merchant_words.col("Label").as("label"));
			
			try {
//				model.write().overwrite().save("/Users/nishantarora/Downloads/w2vModel");
				hashingTF.write().overwrite().save("/Users/nishantarora/Downloads/docHash");
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			try {
//				model.write().overwrite().save("/Users/nishantarora/Downloads/w2vModel");
				idfModel.write().overwrite().save("/Users/nishantarora/Downloads/docIDF");
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
					
			Dataset<Row> trainingDF = Location_words.union(Merchant_words);
//			trainingDF.show(false);
			
			DecisionTreeClassifier dt = new DecisionTreeClassifier()
					  .setLabelCol("label")
					  .setFeaturesCol("features")
					  .setSeed(1L);
			DecisionTreeClassificationModel gbtModel = dt.fit(trainingDF);
			
//			RandomForestClassifier forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features");
//			RandomForestClassificationModel gbtModel  = forest.fit(trainingDF);
//			GBTClassifier gbt = new GBTClassifier().setLabelCol("label").setFeaturesCol("features").setMaxIter(10);
//			GBTClassificationModel gbtModel = gbt.fit(Location_words);
//			RandomForestClassifier forest = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features");
//			RandomForestClassificationModel forestModel  = forest.fit(trainingDF);
			
			/*Dataset<Row> predictions = gbtModel.transform(testingDF);
//			predictions.show(false);
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()	
					  .setLabelCol("label")
					  .setPredictionCol("prediction")
					  .setMetricName("accuracy");
			double accuracy = evaluator.evaluate(predictions);
			
			
			
			System.out.println("Test set accuracy = " + accuracy + " count = " + predictions.filter(predictions.col("prediction").equalTo(0.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(0.0)).count() + " count = " + predictions.filter(predictions.col("prediction").equalTo(1.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(1.0)).count());
*/
			try {
				gbtModel.write().overwrite().save("/Users/nishantarora/Downloads/docHash");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			/*predictions.createOrReplaceTempView("pred");
			Dataset<Row> filterr = spark.sql("select * from pred where label=0 and prediction=1").toDF();
			filterr.show(200);*/
			
					
			
		//	Double x = dtModel.predict((Vector) words.select("features"));
			//System.out.println(x);
			
			
//			Dataset<Row> predict=dtModel.transform(hashed);
//			predict.show(false);
			
			
			
			
	}

}
