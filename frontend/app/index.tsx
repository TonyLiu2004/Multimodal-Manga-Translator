import { useWindowDimensions, Text, ScrollView, View, StyleSheet, ActivityIndicator, TextInput, Image } from "react-native";
import React, { useEffect, useState } from 'react';

const BASE_URL = "https://api.mangadex.org";

export default function Index() {
  const [searchQuery, setSearchQuery] = useState('');
  const [mangaList, setMangaList] = useState<any[]>([]);

  const { width } = useWindowDimensions();
  const isDesktop = width > 600;


  const handleSearch = async () => {
    console.log("Searching for:", searchQuery);
    if (!searchQuery) return;
    try {
      const response = await fetch(`https://api.mangadex.org/manga?title=${searchQuery}&limit=10&includes[]=cover_art`);
      const json = await response.json();
      setMangaList(json.data || []);
      console.log("Search results:", json.data);
    } catch (error) {
      console.error("Search error:", error);
    }
  };

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, alignItems: 'center', padding: 20, justifyContent: 'center' }}>
      <Text style={styles.h1}>Manglify</Text>
      <TextInput
        style={styles.input}
        onChangeText={(query) => setSearchQuery(query)}
        onSubmitEditing={handleSearch}
        value={searchQuery}
        placeholder="Search manga title"
      />
      
      {mangaList.length > 0 && (
        <View style={{ marginTop: 20, alignItems: 'center' }}>
          <Text>{mangaList.length} manga found!</Text>
          <View style={{ marginTop: 20, alignItems: 'center' }}>
            {mangaList.map((manga) => {
              const titles = Object.values(manga.attributes.title);
              const displayTitle = titles[0] as string || "Untitled";

              // get cover art url
              const coverArt = manga.relationships.find((rel: any) => rel.type === 'cover_art');
              const fileName = coverArt?.attributes?.fileName;
              const coverUrl = fileName 
                ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
                : 'https://via.placeholder.com/256x360?text=No+Cover';
              return (
                <View key={manga.id} style={{ 
                  marginBottom: 10, 
                  justifyContent: 'flex-start',
                  flexDirection: isDesktop ? 'row' : 'column',
                  width: '50%',
                }}>
                  <Image 
                    source={{ uri: coverUrl }} 
                    style={{ width: 256, height: 360 }}
                  />
                  <View style={{ flex: 1, marginLeft: 30 }}>
                    <Text style={{ fontWeight: 'bold' }}>{displayTitle}</Text>
                    <Text style={{ color: 'gray' }}>ID: {manga.id}</Text>
                    <Text>{manga.attributes.description.en || "No description available."}</Text>
                  </View>
                </View>
              );
            })}
          </View>
        </View>
      )}

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  h1: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#333',
    marginVertical: 10,
  },
  input: {
    height: 50,
    width: '50%',
    borderWidth: 1,
    borderColor: '#E0E0E0', // Light border color
    borderRadius: 10,
    paddingHorizontal: 15,
    fontSize: 16,
    color: '#333',          // Color of the text YOU type
    backgroundColor: '#FAFAFA',
  },
});