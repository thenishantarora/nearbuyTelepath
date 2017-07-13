import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDFModel;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.json.simple.JSONObject;
import scala.collection.mutable.WrappedArray;


public class predictorForAPI {
	
	public static void main(String[] args) {
			
		// TODO Auto-generated method stub
		
		// Loading the Spark Configuration
			SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
			JavaSparkContext sc = new JavaSparkContext(conf);
			SparkSession ssc = new SparkSession(sc.sc());
			SQLContext spark = ssc.sqlContext();
			

		// Loading the Random Forest Classification Model	
			RandomForestClassificationModel dtModel = RandomForestClassificationModel.load("/Users/nishantarora/Downloads/rfFinal");

			String x= "{'query': 'Pizza in Mayur Vihar '}";
			
			
			List<String> list = new ArrayList<String>();
			list.add(x);
			
			JavaRDD<String> inputRDD = sc.parallelize(list).map(new Function<String, String>() {

				@Override
				public String call(String v1) throws Exception {
					return v1;
				}
			});
			
			Dataset<Row> string = spark.read().json(inputRDD);
			string.show();
		// Taking the input string to be predicted
			//Dataset<Row> string = spark.read().text("/Users/nishantarora/Downloads/string.txt");
			
		// Tokenizing the string	
			Tokenizer tokenizer = new Tokenizer().setInputCol("query").setOutputCol("tokens");
			Dataset<Row> tokens = tokenizer.transform(string);
		// Removing the Stop Words	
			StopWordsRemover swr = new StopWordsRemover().setInputCol("tokens").setOutputCol("words");
			Dataset<Row> words = swr.transform(tokens);
		// Forming Unigrams and Bigrams	and adding a length column to the dataframe, len=1 for unigrams and len=2 for bigrams
			NGram unigram = new NGram().setN(1).setInputCol("words").setOutputCol("grams");
			NGram bigram = new NGram().setN(2).setInputCol("words").setOutputCol("grams");
			Dataset<Row> words1 = unigram.transform(words);
			words1 = words1.withColumn("len", functions.lit(1));
			Dataset<Row> words2 = bigram.transform(words);
			words2 = words2.withColumn("len", functions.lit(2));
			words = words1.union(words2);
			
		// Merging the unigrams and bigrams in a same column and then wrapping them inside arrays	
			Dataset<Row> explode = words.withColumn("bi", org.apache.spark.sql.functions.explode(words.col("grams")));
			words = explode.select("bi","len");
			Dataset<Row> finalWords = words.groupBy("bi","len").agg(org.apache.spark.sql.functions.collect_list("bi").as("Words")).select("Words","len");

		// Sorting the final dataframe based upon the length column so that bigrams are always above unigrams in the dataframe	
			finalWords = finalWords.sort(org.apache.spark.sql.functions.desc("len"));
			
		// Loading the HashingTF and the IDF Models	
			HashingTF hash = HashingTF.load("/Users/nishantarora/Downloads/RFhashModel");
			IDFModel idf = IDFModel.load("/Users/nishantarora/Downloads/RFidfModel");

		// Using TF-IDF to form features	
			Dataset<Row> Final = hash.transform(finalWords);
			Final = idf.transform(Final);
			
		// Using Word2Vec to form features	
			Word2VecModel model = Word2VecModel.load("/Users/nishantarora/Downloads/rfW2VModel");
			Final = model.transform(Final);

		// Using Vector Assembler to combine the features formed by TF-IDF and Word2Vec	
			VectorAssembler assembler = new VectorAssembler()
					  .setInputCols(new String[]{"w2v", "features1"})
					  .setOutputCol("features");
			Final = assembler.transform(Final);	

		// Making the predictions	
			Dataset<Row> predict=dtModel.transform(Final);
			predict = predict.select("Words","probability","prediction");
			
			predict.show(false);
		// Showing the final output	
			
			JSONObject locJson = new JSONObject();
			locJson.put("Category", "location");
			
			JSONObject mercJson = new JSONObject();
			mercJson.put("Category", "merchant");
			
			Boolean locFlag = true, mercFlag = true;
			for(Row r: predict.collectAsList()){
				Vector v= r.getAs(1);
				if(r.getAs(2).equals(0.0) && locFlag)
				{
					if(v.apply(0)>=0.7)
					{
						WrappedArray<String> wa = (WrappedArray<String>) r.get(0);
						String loc = wa.mkString(" ");
						locJson.put("Name", loc);
						locJson.put("Probability", v.apply(0));
						System.out.println("Location is " + loc + " with a probability of " + v.apply(0));
						locFlag = false;
					}
				}
				else if(r.getAs(2).equals(1.0) && mercFlag)
				{
					if(v.apply(1)>=0.7)
					{
						WrappedArray<String> wa = (WrappedArray<String>) r.get(0);
						String merc = wa.mkString(" ");
						mercJson.put("Name", merc);
						mercJson.put("Probability", v.apply(1));
						System.out.println("Merchant is " + merc + " with a probability of " + v.apply(1));
						mercFlag = false;
					}
				}
			}	
			
	}
}
