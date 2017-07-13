import static spark.Spark.*;

import java.util.List;

import com.google.gson.Gson;
import org.json.simple.JSONObject;

 class stringQuery {
 	 
	private String query;
	
	public void setQuery(String s)
	{
		this.query = s;
	}	
}
public class API {

	public static String q="pizza in mayur vihar";

	public static void main(String[] args) {
		// TODO Auto-generated method stub		
        post("/predictEntity", (req, res) -> 
        {
        	q =req.queryParams("q");
        	System.out.println(q);
        	
        	
        	return "OK";
        });
        

        
        Gson gson = new Gson();
        stringQuery obj = new stringQuery();
        obj.setQuery(q);
        String jsonString = gson.toJson(obj);
//        System.out.println(jsonString);
        
        get("/test", (req, res) -> 
        {
        	apiFunc function = new apiFunc();
        	List<NameEntity> nameEntities = function.stringPredictor(jsonString);
        	return gson.toJson(nameEntities);
        });
	}
	
    

    
    

}

