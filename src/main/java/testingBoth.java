import org.apache.commons.beanutils.converters.StringConverter;
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
//import org.apache.spark.util.Vector;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.feature.VectorSlicer;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.ml.feature.SQLTransformer;

import scala.collection.Seq;
import scala.collection.mutable.WrappedArray;

import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IDFModel;
import java.lang.String;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;


public class testingBoth {
	
	public static String stringConvert(String s)
	{
		int l = s.length();
		return s.substring(2, l-2);
	}
	public static String stringConvert2(String s)
	{
		int l = s.length();
		return s.substring(1, l-1);
	}
	

	public static void main(String[] args) {
			
		// TODO Auto-generated method stub
			SparkConf conf = new SparkConf().setAppName("Name").setMaster("local");
			JavaSparkContext sc = new JavaSparkContext(conf);
			SparkSession ssc = new SparkSession(sc.sc());
			SQLContext spark = ssc.sqlContext();
			
//			DecisionTreeClassificationModel dtModel = DecisionTreeClassificationModel.load("/Users/nishantarora/Downloads/hashAndW2VnewData" );
//			GBTClassificationModel dtModel = GBTClassificationModel.load("/Users/nishantarora/Downloads/Final");
			RandomForestClassificationModel dtModel = RandomForestClassificationModel.load("/Users/nishantarora/Downloads/rfFinal");

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
			//words.withColumn("len", functions.lit(1.0));
			words = bigram.transform(words);
//			words.show(false);
			Dataset<Row> explode = words.withColumn("bi", org.apache.spark.sql.functions.explode(words.col("bigrams"))).withColumn("uni", org.apache.spark.sql.functions.explode(words.col("unigrams")));
			explode.show(false);
			Dataset<Row> unigrams = explode.select("uni");
			Dataset<Row> bigrams = explode.select("bi");
			words = bigrams.union(unigrams).distinct();
			SQLTransformer sql = new SQLTransformer().setStatement("SELECT *, length(bi) AS Len FROM words");
			words.createOrReplaceTempView("words");
			words = sql.transform(words);
			Dataset<Row> len = words.select("bi","Len");
			len.show(false);
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
			Dataset<Row> finalWords = words.groupBy("bi").agg(org.apache.spark.sql.functions.collect_list("bi").as("Words"));
			Column c = len.col("Len");
		
//			finalWords = finalWords.select("Words");
			finalWords = finalWords.as("f").join(len.as("l"), "bi").select("f.Words","l.Len");
//			finalWords = finalWords.withColumn("yo", c);
//			finalWords = finalWords.numericColumns();
//			words = words.groupBy("bi").agg(org.apache.spark.sql.functions.collect_list("bi").as("Words"));
//			words.show(false);			
//			finalWords = finalWords.select("Words",len.select("Len").as("ss"));
//			finalWords = finalWords.union(len);	
//			SQLTransformer sql = new SQLTransformer().setStatement("SELECT *, org.apache.spark.sql.functions.length(Words) AS Len FROM finalWords");
//			finalWords.createOrReplaceTempView("finalWords");
//			finalWords = sql.transform(finalWords);
//			finalWords = finalWords.withColumn("len", functions.lit(finalWords.col("Words").toString()));
//			finalWords = finalWords.wit
			finalWords = finalWords.sort(org.apache.spark.sql.functions.desc("Len"));
			finalWords.show(false);
			int numFeatures = 20;
			HashingTF hash = HashingTF.load("/Users/nishantarora/Downloads/RFhashModel");
			IDFModel idf = IDFModel.load("/Users/nishantarora/Downloads/RFidfModel");

			Dataset<Row> Final = hash.transform(finalWords);
			Final = idf.transform(Final);
			
//			hashed = hashed.select(hashed.col("Words"),hashed.col("features1").as("hashs"));
			Word2VecModel model = Word2VecModel.load("/Users/nishantarora/Downloads/rfW2VModel");
			Final = model.transform(Final);
//			Dataset<Row> w2v = model.transform(finalWords).select("Words","w2v");
//			Dataset<Row> w2vecs = w2v.select("Words","w2v");
//			Dataset<Row> union = hashed.union(w2vecs);
//			union.show(false);
			VectorAssembler assembler = new VectorAssembler()
					  .setInputCols(new String[]{"w2v", "features1"})
					  .setOutputCol("features");

			Final = assembler.transform(Final);
//			finalWords.show(false);
			//Word2Vec word2Vec = new Word2Vec().setInputCol("Words").setOutputCol("w2v").setVectorSize(10).setMinCount(0);
			/*Word2VecModel model = Word2VecModel.load("/Users/nishantarora/Downloads/w2vModel");
//			Word2VecModel model = word2Vec.fit(finalWords);
			Dataset<Row> w2v = model.transform(finalWords).select("Words","features");*/
//			words = w2v.select(w2v.col("Words"),w2v.col("features"));
			/*int numFeatures =10;
			HashingTF hashingTF = new HashingTF()
					  .setInputCol("uniArray")
					  .setOutputCol("rawFeatures")
					  .setNumFeatures(numFeatures);

			Dataset<Row> featurizedData = hashingTF.transform(w2v);
					// alternatively, CountVectorizer can also be used to get term frequency vectors

			IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features1");
			IDFModel idfModel = idf.fit(featurizedData);

			Dataset<Row> rescaledData = idfModel.transform(featurizedData);
			words = rescaledData;
			VectorAssembler assembler = new VectorAssembler()
					  .setInputCols(new String[]{"features1", "w2v"})
					  .setOutputCol("features");

			words = assembler.transform(words);
			words = words.select("uniArray","rawFeatures","features");
			words.show(false);*/
			
			
		//	Double x = dtModel.predict((Vector) words.select("features"));
			//System.out.println(x);
			
			
			Dataset<Row> predict=dtModel.transform(Final);
			predict = predict.select("Words","probability","prediction");
			predict.show(false);
			Boolean locFlag = true, mercFlag = true;
			for(Row r: predict.collectAsList()){
				Vector v= r.getAs(1);
				if(r.getAs(2).equals(0.0) && locFlag)
				{
					if(v.apply(0)>=0.7)
					{
						WrappedArray<String> wa = (WrappedArray<String>) r.get(0);
						String loc = wa.mkString(" ");
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
						System.out.println("Merchant is " + merc + " with a probability of " + v.apply(1));
						mercFlag = false;
					}
				}
			}
/*			VectorSlicer vs1 = new VectorSlicer().setInputCol("probability").setOutputCol("first");
			VectorSlicer vs2 = new VectorSlicer().setInputCol("probability").setOutputCol("second");
			vs1.setIndices(new int[]{0});
			vs2.setIndices(new int[]{1});
			predict = vs1.transform(predict);
			predict = vs2.transform(predict);
//			predict = predict.withColumn("1st", org.apache.spark.sql.functions.explode(predict.col("first")));
//			predict = predict.withColumn("2nd", org.apache.spark.sql.functions.explode(predict.col("second")));

//			List<double[]> myList = new ArrayList<double[]>();
//			Object x = predict.collect();
			List<Row> Words = predict.select("Words").collectAsList();
			List<Row> first = predict.select("first").collectAsList();
//			System.out.println(first.size());
			List<Row> second = predict.select("second").collectAsList();

//			Row r = first.get(1);
			
//			System.out.println(r.toString());
			List<Pair<Double,String>> loc = new ArrayList<Pair<Double,String>>();
			List<Pair<Double,String>> merc = new ArrayList<Pair<Double,String>>();

			List<Row> labels = predict.select("prediction").collectAsList();
			System.out.println(labels.get(0).toString() + " " + labels.size());
			for(int i=0;i<first.size();i++)
			{
				String label = stringConvert2(labels.get(i).toString());
				System.out.println(label);
				
				double locProb = Double.parseDouble(stringConvert((first.get(i).toString())));
				double mercProb = Double.parseDouble(stringConvert(second.get(i).toString()));
				int length = Words.get(i).toString().length()-4;
				String word = stringConvert(Words.get(i).toString());
				String ss = "0.0";
				if(label.equals(ss))
				{
					Pair newPair = new Pair(locProb*length,word);
					loc.add(newPair);
				}
				else
				{
					Pair newPair = new Pair(mercProb*length,word);
					merc.add(newPair);
				}
				
			}
			System.out.println(loc.size() + " " + merc.size());
			double max = 0;
			int l=-1;
			for(int i=0;i<loc.size();i++)
			{
				if(loc.get(i).getL()>max)
				{
					max=loc.get(i).getL();
					l=i;
				}
			}
			max = 0;
			int r=-1;
			for(int i=0;i<merc.size();i++)
			{
				if(merc.get(i).getL()>max)
				{
					max=merc.get(i).getL();
					r=i;
				}
			}
			
			System.out.println("Location is " + loc.get(l).getR());
			System.out.println("Merchant is " + merc.get(r).getR());
			
//			Object v = collect.get(1);
			

//			Double[] aa = (Double[])v;
//			v.elements()[0].toString();
//			System.out.println(collect[1]);
//			System.out.println(collect.get(1));
			predict.show(false);
			*/

	
			
			
	}

}
