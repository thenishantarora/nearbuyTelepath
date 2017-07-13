
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
public class Test {

	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SparkSession ssc = new SparkSession(sc.sc());
		SQLContext spark = ssc.sqlContext();
		Dataset<Row> training = spark.read().format("csv").option("header", "true").load("/Users/nishantarora/Downloads/location.csv").distinct();
//		System.out.println(training.count());
		
		Tokenizer tokenizer=new Tokenizer().setInputCol("redemptionAddress_locality_name").setOutputCol("words");				  
		Dataset<Row> output=tokenizer.transform(training);
		StopWordsRemover remover = new StopWordsRemover().setInputCol("words").setOutputCol("words2");
		Dataset<Row> afterStop = remover.transform(output);
		NGram unigram = new NGram().setN(1).setInputCol("words2").setOutputCol("Unigrams");
		Dataset<Row> UniGrams=unigram.transform(afterStop);
		
		/*VectorSlicer vectorSlicer = new VectorSlicer().setInputCol("Unigrams").setOutputCol("unigramsSliced");
		vectorSlicer.setIndices(new int[]{0});
		Dataset<Row> test = vectorSlicer.transform(UniGrams);
		UniGrams.show(false);*/
//		UniGrams.printSchema();
//		UniGrams.withColumn("Unigrams",UniGrams.col("Unigrams").cast(DataTypes.StringType));
		
//		UniGrams.selectExpr("redemptionAddress_locality_name","words","words2","split(Unigrams, ',')[0] as Unigrams1","split(Unigrams, ',')[1] as Unigrams2");
		NGram bigram = new NGram().setN(2).setInputCol("words2").setOutputCol("Bigrams");
		Dataset<Row> BiGrams = bigram.transform(afterStop);
//		Dataset<Row> Grams = UniGrams.as("d1").join(BiGrams.as("d2"),"redemptionAddress_locality_name").select("d1.*","d2.Bigrams");

		Dataset<Row> Grams = UniGrams.as("d1").join(BiGrams.as("d2"),"redemptionAddress_locality_name").select("d1.*","d2.Bigrams");
		
		
//		Dataset<Row> explodeGrams = Grams.withColumn("uni", org.apache.spark.sql.functions.explode(Grams.col("Unigrams"))).withColumn("bi", org.apache.spark.sql.functions.explode(Grams.col("Bigrams")));
		Dataset<Row> uni = Grams.select("Unigrams");
		Dataset<Row> bi = Grams.select("Bigrams");
		//Dataset<Row> finalGrams= uni.union(bi).distinct().filter(org.apache.spark.sql.functions.col("uni").startsWith("south"));
		Dataset<Row> finalGrams= uni.union(bi).distinct();
//		finalGrams.show(false);
//		Dataset<Row> test = Grams.select(org.apache.spark.sql.functions.concat(Grams.col("Unigrams"),Grams.col("Bigrams")));
//		Dataset<Row> test = Grams.withColumn("combined", org.apache.spark.sql.functions.array("Unigrams","Bigrams"));
//		test.show(false);
		Word2Vec word2Vec = new Word2Vec().setInputCol("Unigrams").setOutputCol("Vectors").setVectorSize(3).setMinCount(0);
		Word2VecModel model = word2Vec.fit(finalGrams);
		Dataset<Row> vector = model.transform(finalGrams).select("Unigrams","Vectors");  
		vector = vector.withColumn("Label", functions.lit(0.0));
		
//		vector.show(false);
		vector = vector.select(vector.col("Unigrams").as("Grams"),vector.col("Vectors").as("features"),vector.col("Label").as("label"));
		double[] array = {0.7,0.2,0.1};
		Dataset<Row>[] splitting = vector.randomSplit(array, 5065);
		Dataset<Row> Training = splitting[0];
		Dataset<Row> Testing  = splitting[1];
		Dataset<Row> Validation = splitting[2];
		LogisticRegression lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8);
		LogisticRegressionModel lrModel = lr.fit(Training);
		//Training.show(false);
		System.out.println("Coefficients: "+ lrModel.coefficients() + " Intercept: " + lrModel.intercept());
		
		Dataset<Row> predictions = lrModel.transform(Testing);
				//As you can see, the previous model transform produced a new columns: rawPrediction, probablity and prediction.**
		predictions.show(false);
//		Validation.show(false);
		/*Word2Vec word2Vec2 = new Word2Vec().setInputCol("Bigrams").setOutputCol("bigramVectors").setVectorSize(3).setMinCount(0);
		model = word2Vec2.fit(vector);
		Dataset<Row> finalVectors = model.transform(vector).select("Unigrams","Bigrams","unigramVectors","bigramVectors");*/
		//UniGrams.show(false);

//		finalVectors.show(false);
		
//		output.show();
		
	}

}
