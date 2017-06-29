
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
public class Location {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
		// For Location
		Dataset<Row> Location_training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/location.csv").distinct();
//		System.out.println(training.count());
		Tokenizer Location_tokenizer=new Tokenizer().setInputCol("redemptionAddress_locality_name").setOutputCol("words");				  
		Dataset<Row> Location_output=Location_tokenizer.transform(Location_training);
		StopWordsRemover Location_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Location_afterStop = Location_remover.transform(Location_output);
		NGram Location_unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> Location_UniGrams=Location_unigram.transform(Location_afterStop);
		NGram bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> Location_BiGrams = bigram.transform(Location_afterStop);
		Dataset<Row> Location_uni = Location_UniGrams.select("Unigrams");
		Dataset<Row> Location_bi = Location_BiGrams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> Location_finalGrams= Location_uni.union(Location_bi).distinct();
		Word2Vec Location_word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(3).setMinCount(0);
		Word2VecModel Location_model = Location_word2Vec.fit(Location_finalGrams);
		Dataset<Row> Location_vector = Location_model.transform(Location_finalGrams).select("Unigrams","Vectors");  
		Location_vector = Location_vector.withColumn("Label", functions.lit(0.0));		
		Location_vector = Location_vector.select(Location_vector.col("Unigrams").as("Grams"),Location_vector.col("Vectors").as("features"),Location_vector.col("Label").as("label"));
		double[] array = {0.7,0.2,0.1};
		Dataset<Row>[] Location_splitting = Location_vector.randomSplit(array, 5065);
		Dataset<Row> Location_Training = Location_splitting[0];
		Dataset<Row> Location_Testing  = Location_splitting[1];
		Dataset<Row> Location_Validation = Location_splitting[2];
		
		//For Merchant
		
		
		LogisticRegression lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8);
		LogisticRegressionModel lrModel = lr.fit(Location_Training);
		//Training.show(false);
		System.out.println("Coefficients: "+ lrModel.coefficients() + " Intercept: " + lrModel.intercept());		
		Dataset<Row> predictions = lrModel.transform(Location_Testing);
		predictions.show(false);

		
	}

}
