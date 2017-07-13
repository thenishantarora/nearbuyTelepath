
import java.io.IOException;
import java.util.ArrayList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
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
public class hashingW2Vfinal {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local[4]");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
		// For Location
		Dataset<Row> training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/allData.csv");
		/*Dataset<Row> data = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/data.csv");
		Dataset<Row> ld1 = data.select(data.col("locality_detail1").as("loc")).distinct();
		Dataset<Row> ld2 = data.select(data.col("locality_detail2").as("loc")).distinct();
		Dataset<Row> ld3 = data.select(data.col("locality_detail3").as("loc")).distinct();*/

//		System.out.println(training.count());
		// For Location
		
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
		System.out.println(Location_training.count());
		Dataset<Row> Location_1 = Location_training;
		Dataset<Row> Location_2 = Location_training;
		Dataset<Row> Location_4 = Location_training;

		Location_training = Location_training.union(Location_1).union(Location_2).union(Location_4);
		Location_training = Location_training.filter(Location_training.col("loc").isNotNull());
//		Location_training = Location_training.union(Location_training3).un.distinct();
		Tokenizer Location_tokenizer=new Tokenizer().setInputCol("loc").setOutputCol("words");				  
		Dataset<Row> Location_output=Location_tokenizer.transform(Location_training);
		StopWordsRemover Location_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Location_afterStop = Location_remover.transform(Location_output);
		/*Dataset<Row> explode1 = Location_afterStop.withColumn("one", org.apache.spark.sql.functions.explode(Location_afterStop.col("words2")));
//		System.out.println(explode.select("one").count() + " " + Location_afterStop.select("words2").count());
		explode1 = explode1.groupBy("one").agg(org.apache.spark.sql.functions.collect_list("one").as("uniArray"));
		Location_afterStop = explode1;*/
		/*Word2Vec Location_word2Vec = new Word2Vec().setInputCol("words2").setOutputCol("features").setVectorSize(8).setMinCount(0);
		Word2VecModel Location_model = Location_word2Vec.fit(Location_afterStop);
		Dataset<Row> Location_w2v = Location_model.transform(Location_afterStop).select("words2","features");*/
//		explode.show(false);
//		Location_afterStop.show(false);
		/*int numFeatures = 10;
		HashingTF Location_hashingTF = new HashingTF()
				  .setInputCol("words2")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> Location_featurizedData = Location_hashingTF.transform(Location_w2v);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

		IDF Location_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
		IDFModel Location_idfModel = Location_idf.fit(Location_featurizedData);

		Dataset<Row> Location_rescaledData = Location_idfModel.transform(Location_featurizedData);*/
		Dataset<Row> Location_vector = Location_afterStop;

		
		

		Location_vector = Location_vector.withColumn("Label", functions.lit(0.0));		
		Location_vector = Location_vector.select(Location_vector.col("words2").as("Words"),Location_vector.col("Label").as("label"));
		//System.out.println(Location_vector.count());

		//For Merchant
		
//		Dataset<Row> Merchant_training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/merchants.csv").distinct();
		Dataset<Row> Merchant_training = training.select("name").distinct();
//		Merchant_training = Merchant_training.union(temp).distinct();
		//System.out.println(Merchant_training.count());
		Tokenizer Merchant_tokenizer=new Tokenizer().setInputCol("name").setOutputCol("words");				  
		Dataset<Row> Merchant_output=Merchant_tokenizer.transform(Merchant_training);
		StopWordsRemover Merchant_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Merchant_afterStop = Merchant_remover.transform(Merchant_output);
		/*Dataset<Row> explode = Merchant_afterStop.withColumn("one", org.apache.spark.sql.functions.explode(Merchant_afterStop.col("words2")));
//		System.out.println(explode.select("one").count() + " " + Location_afterStop.select("words2").count());
		explode = explode.groupBy("one").agg(org.apache.spark.sql.functions.collect_list("one").as("uniArray"));
		Merchant_afterStop = explode;*/
		/*Word2Vec Merchant_word2Vec = new Word2Vec().setInputCol("words2").setOutputCol("features").setVectorSize(8).setMinCount(0);
		Word2VecModel Merchant_model = Merchant_word2Vec.fit(Merchant_afterStop);
		Dataset<Row> Merchant_w2v = Merchant_model.transform(Merchant_afterStop).select("words2","features");*/
		
		/*HashingTF Merchant_hashingTF = new HashingTF()
				  .setInputCol("words2")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> Merchant_featurizedData = Merchant_hashingTF.transform(Merchant_w2v);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

		IDF Merchant_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
		IDFModel Merchant_idfModel = Merchant_idf.fit(Merchant_featurizedData);

		Dataset<Row> Merchant_rescaledData = Merchant_idfModel.transform(Merchant_featurizedData);*/
		Dataset<Row> Merchant_vector = Merchant_afterStop;
		

		Merchant_vector = Merchant_vector.withColumn("Label", functions.lit(1.0));		
		Merchant_vector = Merchant_vector.select(Merchant_vector.col("words2").as("Words"),Merchant_vector.col("Label").as("label"));
		
		/*Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(array, 5065);
		Dataset<Row> Merchant_Training = Merchant_splitting[0];
		Dataset<Row> Merchant_Testing  = Merchant_splitting[1];
		Dataset<Row> Merchant_Validation = Merchant_splitting[2];
		*/
		/*double[] arr = {.12,.88};
		Dataset<Row>[] splitting = Location_vector.randomSplit(arr);
		Location_vector = splitting[0];*/
		
		Dataset<Row> Combined_vector = Location_vector.union(Merchant_vector);
//		Combined_vector.show(false);
		
		NGram unigram = new NGram().setN(1).setInputCol("Words").setOutputCol("grams");
		NGram bigram = new NGram().setN(2).setInputCol("Words").setOutputCol("grams");
		Dataset<Row> Combined_vector1 = unigram.transform(Combined_vector);
		Dataset<Row> Combined_vector2 = bigram.transform(Combined_vector);
//		words.show(false);
//		Dataset<Row> explode = Combined_vector.withColumn("uni", org.apache.spark.sql.functions.explode(Combined_vector.col("unigrams"))).withColumn("bi", org.apache.spark.sql.functions.explode(Combined_vector.col("bigrams")));
//		explode.show(false);
//		Dataset<Row> unigrams = Combined_vector.select("uni","label");
//		Dataset<Row> bigrams = explode.select("bi","label");
		Combined_vector = Combined_vector1.union(Combined_vector2);
//		Combined_vector = Combined_vector.select(Combined_vector.col("uni").as("Words"),Combined_vector.col("label"));
		Combined_vector = Combined_vector.select(Combined_vector.col("grams").as("Words"),Combined_vector.col("label"));
		Word2Vec word2Vec = new Word2Vec().setInputCol("Words").setOutputCol("w2v").setVectorSize(10).setMinCount(0);
		Word2VecModel model = word2Vec.fit(Combined_vector);
		Dataset<Row> w2v = model.transform(Combined_vector).select("Words","w2v","label");
//		w2v.show(false);
		try {
			model.write().overwrite().save("/Users/nishantarora/Downloads/w2v");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		int numFeatures = 20;
		HashingTF hashingTF = new HashingTF()
				  .setInputCol("Words")
				  .setOutputCol("features1")
				  .setNumFeatures(numFeatures);

		Dataset<Row> featurizedData = hashingTF.transform(w2v);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

//		IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
//		IDFModel idfModel = idf.fit(featurizedData);

//		Dataset<Row> rescaledData = idfModel.transform(featurizedData);
//		Combined_vector = rescaledData;
		Combined_vector = featurizedData;
		
		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(new String[]{"w2v", "features1"})
				  .setOutputCol("features");

		Combined_vector = assembler.transform(Combined_vector);
		Combined_vector = Combined_vector.select("Words","features","label");
		try {
//			model.write().overwrite().save("/Users/nishantarora/Downloads/w2vModel");
			hashingTF.write().overwrite().save("/Users/nishantarora/Downloads/hash");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		/*try {
//			model.write().overwrite().save("/Users/nishantarora/Downloads/w2vModel");
			idfModel.write().overwrite().save("/Users/nishantarora/Downloads/idfModel");
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}*/
		double[] fifty = {0.9,0.1};
		Dataset<Row>[] splitting = Combined_vector.randomSplit(fifty);
		Dataset<Row> trainingDF = splitting[0];
		Dataset<Row> testingDF  = splitting[1];
		
		/*Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(fifty);
		Dataset<Row> Merchant_trainingDF = Merchant_splitting[0];
		Dataset<Row> Merchant_testingDF  = Merchant_splitting[1];

		Dataset<Row> trainingDF = Location_trainingDF.union(Merchant_trainingDF);
		Dataset<Row> testingDF  = Location_testingDF.union(Merchant_testingDF);*/
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
			gbtModel.write().overwrite().save("/Users/nishantarora/Downloads/hashW2Vfinal");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}


		
	

}
