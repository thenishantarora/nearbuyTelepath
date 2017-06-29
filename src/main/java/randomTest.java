
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.io.BufferedWriter;
import java.io.FileWriter;
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
import org.omg.PortableInterceptor.LOCATION_FORWARD;
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
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
public class randomTest {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		final String filename = "/Users/nishantarora/Downloads/file.txt";
		String a="";
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
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
//		System.out.println(Location_training.count());
//		Location_training = Location_training.union(Location_training3).un.distinct();
		Tokenizer Location_tokenizer=new Tokenizer().setInputCol("loc").setOutputCol("words");				  
		Dataset<Row> Location_output=Location_tokenizer.transform(Location_training);
		StopWordsRemover Location_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Location_afterStop = Location_remover.transform(Location_output);
		Word2Vec Location_word2Vec = new Word2Vec().setInputCol("words2").setOutputCol("w2v").setVectorSize(4).setMinCount(0);
		Word2VecModel Location_model = Location_word2Vec.fit(Location_afterStop);
		Dataset<Row> Location_w2v = Location_model.transform(Location_afterStop).select("words2","w2v");
		
			
		int numFeatures =10 ;
		HashingTF Location_hashingTF = new HashingTF()
				  .setInputCol("words2")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> Location_featurizedData = Location_hashingTF.transform(Location_w2v);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

//		IDF Location_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
//		IDFModel Location_idfModel = Location_idf.fit(Location_featurizedData);

//		Dataset<Row> Location_rescaledData = Location_idfModel.transform(Location_featurizedData);
		Dataset<Row> Location_vector = Location_featurizedData;
		VectorAssembler Location_assembler = new VectorAssembler()
				  .setInputCols(new String[]{"rawFeatures", "w2v"})
				  .setOutputCol("features");

		Location_vector = Location_assembler.transform(Location_vector);
		Location_vector = Location_vector.select("words2","features");
//		Location_vector.show(false);
		
		

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
		Location_vector = Location_vector.select(Location_vector.col("words2").as("Words"),Location_vector.col("features"),Location_vector.col("Label").as("label"));

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
		Merchant_training = Merchant_training.union(temp);
//		System.out.println(Merchant_training.count());
		Tokenizer Merchant_tokenizer=new Tokenizer().setInputCol("name").setOutputCol("words");				  
		Dataset<Row> Merchant_output=Merchant_tokenizer.transform(Merchant_training);
		StopWordsRemover Merchant_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Merchant_afterStop = Merchant_remover.transform(Merchant_output);
		Word2Vec Merchant_word2Vec = new Word2Vec().setInputCol("words2").setOutputCol("w2v").setVectorSize(4).setMinCount(0);
		Word2VecModel Merchant_model = Merchant_word2Vec.fit(Merchant_afterStop);
		Dataset<Row> Merchant_w2v = Merchant_model.transform(Merchant_afterStop).select("words2","w2v");
		
		HashingTF Merchant_hashingTF = new HashingTF()
				  .setInputCol("words2")
				  .setOutputCol("rawFeatures")
				  .setNumFeatures(numFeatures);

		Dataset<Row> Merchant_featurizedData = Merchant_hashingTF.transform(Merchant_w2v);
				// alternatively, CountVectorizer can also be used to get term frequency vectors

//		IDF Merchant_idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
//		IDFModel Merchant_idfModel = Merchant_idf.fit(Merchant_featurizedData);

//		Dataset<Row> Merchant_rescaledData = Merchant_idfModel.transform(Merchant_featurizedData);
		Dataset<Row> Merchant_vector = Merchant_featurizedData;
		VectorAssembler Merchant_assembler = new VectorAssembler()
				  .setInputCols(new String[]{"rawFeatures", "w2v"})
				  .setOutputCol("features");

		Merchant_vector = Merchant_assembler.transform(Merchant_vector);
		Merchant_vector = Merchant_vector.select("words2","features");
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
		Merchant_vector = Merchant_vector.select(Merchant_vector.col("words2").as("Words"),Merchant_vector.col("features"),Merchant_vector.col("Label").as("label"));
		
		/*Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(array, 5065);
		Dataset<Row> Merchant_Training = Merchant_splitting[0];
		Dataset<Row> Merchant_Testing  = Merchant_splitting[1];
		Dataset<Row> Merchant_Validation = Merchant_splitting[2];
		*/
		/*double[] arr = {.24,.76};
		Dataset<Row>[] splitting = Merchant_vector.randomSplit(arr);
		Merchant_vector = splitting[0];*/
		double[] arrrr = {.12,.88};
		Dataset<Row>[] splitting = Location_vector.randomSplit(arrrr, 100000L);
		Location_vector = splitting[0];
		
//		double[] array = {0.7,0.2,0.1};
		/*double [] m = {0.05,0.95};
		Dataset<Row>[] merchantSplit  = Merchant_vector.randomSplit(m);
		Merchant_vector = merchantSplit[0];*/
		/*Dataset<Row>[] Location_splitting = Location_vector.randomSplit(array);
		Dataset<Row> Location_trainingDF = Location_splitting[0];
		Dataset<Row> Location_testingDF  = Location_splitting[1];
		Dataset<Row> Location_validationDF = Location_splitting[2];
		
		Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(array);
		Dataset<Row> Merchant_trainingDF = Merchant_splitting[0];
		Dataset<Row> Merchant_testingDF  = Merchant_splitting[1];
		Dataset<Row> Merchant_validationDF = Merchant_splitting[2];*/
//		Dataset<Row> mergedDF =  Location_vector.union(Merchant_vector);
		
		/*MinMaxScaler scaler = new MinMaxScaler().setInputCol("features").setOutputCol("scaledFeatures");
		MinMaxScalerModel scalerModel = scaler.fit(mergedDF);
		mergedDF = scalerModel.transform(mergedDF);
		mergedDF = mergedDF.select(mergedDF.col("Grams"),mergedDF.col("scaledFeatures").as("features"),mergedDF.col("label"));*/
//		Dataset<Row>[] merged_splitting = mergedDF.randomSplit(array);
		List<Long> seeds = new ArrayList<Long>();
		seeds.add(35561L);
		seeds.add(67683L);
		seeds.add(24251L);
		seeds.add(99383L);

		DecisionTreeClassificationModel dtModel=null;
		Dataset<Row> testingDF=null;
		for(Long l : seeds)
		{
			double[] fifty = {0.8,0.2};
			Dataset<Row>[] Location_splitting = Location_vector.randomSplit(fifty,l);
			Dataset<Row> Location_trainingDF = Location_splitting[0];
			Dataset<Row> Location_testingDF  = Location_splitting[1];
			
			Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(fifty,l);
			Dataset<Row> Merchant_trainingDF = Merchant_splitting[0];
			Dataset<Row> Merchant_testingDF  = Merchant_splitting[1];
			Dataset<Row> trainingDF = Location_trainingDF.union(Merchant_trainingDF);
			testingDF  = Location_testingDF.union(Merchant_testingDF);
	//		Dataset<Row> validationDF = Location_validationDF.union(Merchant_validationDF);
			
			DecisionTreeClassifier dt = new DecisionTreeClassifier()
					  .setLabelCol("label")
					  .setFeaturesCol("features").setSeed(1L);
					 
			dtModel = dt.fit(trainingDF);
		
		}
		
		/*NaiveBayes nb = new NaiveBayes();		
			NaiveBayesModel model = nb.fit(trainingDF);*/
			Dataset<Row> predictions = dtModel.transform(testingDF);
	//		predictions.show(false);
			MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
					  .setLabelCol("label")
					  .setPredictionCol("prediction")
					  .setMetricName("accuracy");
			double accuracy = evaluator.evaluate(predictions);
			
			
			
			
//			String a ="Test set accuracy = " + accuracy + " count = " + predictions.filter(predictions.col("prediction").equalTo(0.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(0.0)).count() + " count = " + predictions.filter(predictions.col("prediction").equalTo(1.0)).count() + " " + predictions.filter(predictions.col("label").equalTo(1.0)).count() + "\n";
			 a = "Number of features= "+numFeatures+" Test set accuracy = " + accuracy+"\n";
			//		predictions.filter(predictions.col("prediction").equalTo(0.0)).count();
			
			String b = "Learned classification tree model:\n" + dtModel.toDebugString() + "\n";

			System.out.println(a);
			System.out.println(b);
	//		predictions.filter(predictions.col("label").equalTo(1.0)).show(200);
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
	
			try {
				dtModel.write().overwrite().save("/Users/nishantarora/Downloads/randomTest");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			

	}

}
