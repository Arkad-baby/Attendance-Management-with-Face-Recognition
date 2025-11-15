from SupabaseClient import supabase


response=supabase.table("Users").select("Username", "Embeddings").execute()

for row in response.data:
    print(row["Username"])

