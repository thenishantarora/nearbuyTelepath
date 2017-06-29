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
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import java.lang.String;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.NaiveBayesModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;
public class testString2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
			SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
			JavaSparkContext sc = new JavaSparkContext(conf);
			SparkSession ssc = new SparkSession(sc.sc());
			SQLContext spark = ssc.sqlContext();
			
//			RandomForestClassificationModel dtModel = RandomForestClassificationModel.load("/Users/nishantarora/Downloads/randomForest");
//			NaiveBayesModel dtModel = NaiveBayesModel.load("/Users/nishantarora/Downloads/NaiveBayes");
			GBTClassificationModel dtModel = GBTClassificationModel.load("/Users/nishantarora/Downloads/gbTree");
//			System.out.println(dtModel.toDebugString());
//			OneVsRestModel dtModel = OneVsRestModel.load("/Users/nishantarora/Downloads/oneVsAll");
			String s = " ";
			Dataset<Row> string = spark.read().text("/Users/nishantarora/Downloads/string.txt");
//			string.show(false);
			Tokenizer tokenizer = new Tokenizer().setInputCol("value").setOutputCol("tokens");
			Dataset<Row> tokens = tokenizer.transform(string);
//			tokens.show(false);
			StopWordsRemover swr = new StopWordsRemover().setInputCol("tokens").setOutputCol("words");
			Dataset<Row> words = swr.transform(tokens);
//			words.show(false);
			NGram unigram = new NGram().setN(1).setInputCol("words").setOutputCol("unigrams");
			NGram bigram = new NGram().setN(2).setInputCol("words").setOutputCol("bigrams");
			words = unigram.transform(words);
			words = bigram.transform(words);
//			words.show(false);
			Dataset<Row> explode = words.withColumn("uni", org.apache.spark.sql.functions.explode(words.col("unigrams"))).withColumn("bi", org.apache.spark.sql.functions.explode(words.col("bigrams")));
//			explode.show(false);
			Dataset<Row> unigrams = explode.select("uni");
			Dataset<Row> bigrams = explode.select("bi");
			words = unigrams.union(bigrams).distinct();
//			words.show(false);
			
			/*JavaRDD<String[]> wordsArray= words.toJavaRDD().map(new Function<Row, String[]>() {

				public String[] call(Row row) throws Exception {
					String r =row.getAs(0);
					String [] x = new String[]{r};
					return x;
				}
			});
			*/
		/*	StructType st1 = new StructType().add("uni", DataTypes.StringType);
			val schema = StructType(Array(StructField("firstName",StringType,true),StructField("lastName",StringType,true),StructField("age",IntegerType,true)))
			Dataset<Row> finalWords = spark.createDataFrame(wordsArray, String.class);
			finalWords.show();
			*/
			Dataset<Row> finalWords = words.groupBy("uni").agg(org.apache.spark.sql.functions.collect_list("uni").as("uniArray"));
//			finalWords.show(false);
//			Word2Vec word2Vec = new Word2Vec().setInputCol("uniArray").setOutputCol("w2v").setVectorSize(4).setMinCount(0);
//			Word2VecModel model = word2Vec.fit(finalWords);
//			Dataset<Row> w2v = model.transform(finalWords).select("uniArray","w2v");
			int numFeatures = 10;
			HashingTF hashingTF = new HashingTF()
					  .setInputCol("uniArray")
					  .setOutputCol("rawFeatures")
					  .setNumFeatures(numFeatures);

			Dataset<Row> featurizedData = hashingTF.transform(finalWords);
					// alternatively, CountVectorizer can also be used to get term frequency vectors

			IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
			IDFModel idfModel = idf.fit(featurizedData);

			Dataset<Row> rescaledData = idfModel.transform(featurizedData);
			words = rescaledData;
			/*VectorAssembler assembler = new VectorAssembler()
					  .setInputCols(new String[]{"features1", "w2v"})
					  .setOutputCol("features");

			words = assembler.transform(words);*/
			words = words.select("uniArray","features");
			words.show(false);
			
			
		//	Double x = dtModel.predict((Vector) words.select("features"));
			//System.out.println(x);
			
			
			Dataset<Row> predict=dtModel.transform(words);
			predict.show(false);
			
			
			
			
			
	}

}
