
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
public class Merchant {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		
		// For Location
		Dataset<Row> Merchant_training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/merchants.csv").distinct();
//		System.out.println(training.count());
		Tokenizer Merchant_tokenizer=new Tokenizer().setInputCol("name").setOutputCol("words");				  
		Dataset<Row> Merchant_output=Merchant_tokenizer.transform(Merchant_training);
		StopWordsRemover Merchant_remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> Merchant_afterStop = Merchant_remover.transform(Merchant_output);
		NGram Merchant_unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> Merchant_UniGrams=Merchant_unigram.transform(Merchant_afterStop);
		NGram bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> Merchant_BiGrams = bigram.transform(Merchant_afterStop);
		Dataset<Row> Merchant_uni = Merchant_UniGrams.select("Unigrams");
		Dataset<Row> Merchant_bi = Merchant_BiGrams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> Merchant_finalGrams= Merchant_uni.union(Merchant_bi).distinct();
		Word2Vec Merchant_word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(3).setMinCount(0);
		Word2VecModel Merchant_model = Merchant_word2Vec.fit(Merchant_finalGrams);
		Dataset<Row> Merchant_vector = Merchant_model.transform(Merchant_finalGrams).select("Unigrams","Vectors");  
		Merchant_vector = Merchant_vector.withColumn("Label", functions.lit(1.0));		
		Merchant_vector = Merchant_vector.select(Merchant_vector.col("Unigrams").as("Grams"),Merchant_vector.col("Vectors").as("features"),Merchant_vector.col("Label").as("label"));
		double[] array = {0.7,0.2,0.1};
		Dataset<Row>[] Merchant_splitting = Merchant_vector.randomSplit(array, 5065);
		Dataset<Row> Merchant_Training = Merchant_splitting[0];
		Dataset<Row> Merchant_Testing  = Merchant_splitting[1];
		Dataset<Row> Merchant_Validation = Merchant_splitting[2];
		
		//For Merchant
		
		
		LogisticRegression lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8);
		LogisticRegressionModel lrModel = lr.fit(Merchant_Training);
		//Training.show(false);
		System.out.println("Coefficients: "+ lrModel.coefficients() + " Intercept: " + lrModel.intercept());		
		Dataset<Row> predictions = lrModel.transform(Merchant_Testing);
		predictions.show(false);

		
	}

}
