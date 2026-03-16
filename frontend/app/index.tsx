import { useWindowDimensions, Text, ScrollView, View, StyleSheet, ActivityIndicator, TextInput, Image, Pressable, FlatList, NativeSyntheticEvent, NativeScrollEvent } from "react-native";
import React, { useEffect, useState, useRef } from 'react';
import PopUp from './components/PopUp';
import Carousel, { ICarouselInstance, Pagination } from 'react-native-reanimated-carousel'
import { useSharedValue } from "react-native-reanimated";

const BASE_URL = "https://api.mangadex.org";
const BACKEND_URL = 'https://3ee7-2600-1017-a410-6e3e-8562-f8d-b602-91ea.ngrok-free.app';

interface Chapter {
  id: string;
  chapter: string;
  title: string;
  pages: number;
}

export default function Index() {
  const [searchQuery, setSearchQuery] = useState('');
  const [mangaList, setMangaList] = useState<any[]>([]);
  const [popularManga, setPopularManga] = useState<any[]>([]);
  const [popupVisible, setPopupVisible] = useState(false);
  const [selectedManga, setSelectedManga] = useState<{ title: string; summary: string; coverUrl: string; mangaId: string }>({ title: '', summary: '', coverUrl: '', mangaId: '' });
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);

  const { width } = useWindowDimensions();
  const isDesktop = width > 600;
  const ref = useRef<ICarouselInstance>(null);
  const progress = useSharedValue<number>(0);

  useEffect(() => {
    fetchPopularManga();
  }, []);

  const onPressPagination = (index: number) => {
    ref.current?.scrollTo({
      count: index - progress.value,
      animated: true,
    });
  };

  const handleSearch = async () => {
    console.log("Searching for:", searchQuery);
    if (!searchQuery) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/manga/search?title=${searchQuery}&limit=10&cover_art=true`, {
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const json = await response.json();
      setMangaList(json.data || []);
      console.log("Search results:", json.data);
    } catch (error) {
      console.error("Search error:", error);
      setMangaList([]);
    }
  };

  const fetchPopularManga = async() => {
    try {
      const params = new URLSearchParams({
        limit: "15",
        order_by: 'followedCount',
        order_direction: 'desc',
        cover_art: 'True',
      });
      const response = await fetch(`${BACKEND_URL}/api/manga/search?${params}`, {
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
      });
      
      const json = await response.json();
      setPopularManga(json.data || []);
    } catch (error) {
      console.error("Proxy fetch failed:", error);
      setPopularManga([]);
    }
    // try {
    //   console.log("Fetching popular manga...");
    //   const response = await fetch(`https://api.mangadex.org/manga?limit=15&order[followedCount]=desc&includes[]=cover_art`, {
    //     method: 'GET',
    //     headers: {
    //       'Content-Type': 'application/json',
    //     },
    //   });
      
    //   console.log("Response status:", response.status);
      
    //   if (!response.ok) {
    //     throw new Error(`HTTP error! status: ${response.status}`);
    //   }
      
    //   const data = await response.json();
    //   console.log("Fetched popular manga data:", data.data?.length, "items");
    //   setPopularManga(data.data || []);
    // } catch (error) {
    //   console.error("Can't retrieve popular manga:", error);
    //   setPopularManga([]);
    // }
  };

  const fetchChapters = async (mangaId: string) => {
    setLoadingChapters(true);
    try {
      // const response = await fetch(
      //   `https://api.mangadex.org/manga/${mangaId}/feed?limit=20&order[chapter]=asc&translatedLanguage[]=en`
      // );

      const response = await fetch(`${BACKEND_URL}/api/manga/${mangaId}/chapters?limit=100&order_by=chapter&order_direction=asc&translatedLanguage=en&translatedLanguage=jp`, {
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true',
        },
      });
      const json = await response.json();
      const chapterData: Chapter[] = (json.data || []).map((ch: any) => ({
        id: ch.id,
        chapter: ch.attributes.chapter || '?',
        title: ch.attributes.title || '',
        pages: ch.attributes.pages || 0,
      }));
      setChapters(chapterData);
      console.log(`Fetched ${chapterData.length} chapters`);
    } catch (error) {
      console.error("Error fetching chapters:", error);
      setChapters([]);
    } finally {
      setLoadingChapters(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, alignItems: 'center', padding: 20}}>
      <Text style={styles.h1}>Manglify</Text>
      <TextInput
        style={styles.input}
        onChangeText={(query) => setSearchQuery(query)}
        onSubmitEditing={handleSearch}
        value={searchQuery}
        placeholder="Search manga title"
      />

      <Carousel
        ref={ref}
        data={popularManga}
        width={300}
        height={420}
        autoPlay={true}
        autoPlayInterval={3000}
        loop={true}
        mode="parallax"
        modeConfig={{
          parallaxScrollingScale: 0.85,
          parallaxScrollingOffset: 60,
        }}
        // style={{ width: width - 600, marginTop: 30, paddingHorizontal: (width - 900) / 2}} // On android, the width is different so it collapses when -600.
        style = {{ width: width*0.8, marginTop: 30 }} 
        onProgressChange={(offsetProgress, absoluteProgress) => {
					progress.value = absoluteProgress;
				}}
        renderItem={({ item: manga }) => {
          const titles = Object.values(manga.attributes.title);
          const displayTitle = titles[0] as string || "Untitled";

          // get cover art url
          const coverArt = manga.relationships.find((rel: any) => rel.type === 'cover_art');
          const fileName = coverArt?.attributes?.fileName;
          const coverUrl = fileName 
            ? `https://uploads.mangadex.org/covers/${manga.id}/${fileName}.256.jpg`
            : 'https://via.placeholder.com/256x360?text=No+Cover';
          return (
            <View
              style={{ 
                flex: 1,
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Pressable 
                key={manga.id} 
                style={{ 
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                onPress={() => {
                  setSelectedManga({
                    title: displayTitle,
                    summary: manga.attributes.description.en || "No description available.",
                    coverUrl: coverUrl,
                    mangaId: manga.id
                  });
                  setPopupVisible(true);
                  fetchChapters(manga.id);
                }}
              >
                <Image 
                  source={{ uri: coverUrl }} 
                  style={{ width: 296, height: 420, borderRadius: 10 }}
                />
              </Pressable>
            </View>
          );
        }}
      />

      <Pagination.Basic
        progress={progress}
        data={popularManga}
        dotStyle={{backgroundColor: "black", borderRadius: 50}}
        containerStyle={{gap: 5, marginTop: 10, alignItems: 'center'}}
        onPress={onPressPagination}
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
                <Pressable 
                  key={manga.id} 
                  style={{ 
                    marginBottom: 10, 
                    justifyContent: 'flex-start',
                    flexDirection: isDesktop ? 'row' : 'column',
                    width: '50%',
                  }}
                  onPress={() => {
                    setSelectedManga({
                      title: displayTitle,
                      summary: manga.attributes.description.en || "No description available.",
                      coverUrl: coverUrl,
                      mangaId: manga.id
                    });
                    setPopupVisible(true);
                    fetchChapters(manga.id);
                  }}
                >
                  <Image 
                    source={{ uri: coverUrl }} 
                    style={{ width: 256, height: 360 }}
                  />
                  <View style={{ flex: 1, marginLeft: 30 }}>
                    <Text style={{ fontWeight: 'bold' }}>{displayTitle}</Text>
                    <Text style={{ color: 'gray' }}>ID: {manga.id}</Text>
                    <Text>{manga.attributes.description.en || "No description available."}</Text>
                  </View>
                </Pressable>
              );
            })}
          </View>
        </View>
      )}

      <PopUp 
        visible={popupVisible}
        title={selectedManga.title}
        summary={selectedManga.summary}
        coverArt={selectedManga.coverUrl}
        chapters={chapters}
        loadingChapters={loadingChapters}
        onClose={() => setPopupVisible(false)}
      />

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