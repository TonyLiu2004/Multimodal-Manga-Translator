import { useWindowDimensions, Text, ScrollView, View, StyleSheet, ActivityIndicator, TextInput, Image, Pressable, FlatList, NativeSyntheticEvent, NativeScrollEvent } from "react-native";
import React, { useEffect, useState, useRef } from 'react';
import PopUp from './components/PopUp';
import Carousel, { ICarouselInstance, Pagination } from 'react-native-reanimated-carousel'
import { useSharedValue, Extrapolation, interpolate } from "react-native-reanimated";

const BASE_URL = "https://api.mangadex.org";

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
  const [actionManga, setActionManga] = useState<any[]>([]);
  const [romanceManga, setRomanceManga] = useState<any[]>([]);
  const [recentManga, setRecentManga] = useState<any[]>([]);
  const [popupVisible, setPopupVisible] = useState(false);
  const [selectedManga, setSelectedManga] = useState<{ title: string; summary: string; coverUrl: string; mangaId: string }>({ title: '', summary: '', coverUrl: '', mangaId: '' });
  const [chapters, setChapters] = useState<Chapter[]>([]);
  const [loadingChapters, setLoadingChapters] = useState(false);

  const { width } = useWindowDimensions();
  const isDesktop = width > 600;
  const ref = useRef<ICarouselInstance>(null);
  const progress = useSharedValue<number>(0);

  useEffect(() => {
    async function loadHomePage() {
      const [
        popular,
        action,
        romance,
        recent
      ] = await Promise.all([
        fetchManga("limit=15&order[followedCount]=desc"),
        fetchManga("limit=10&includedTags[]=391b0423-d847-456f-aff0-8b0cfc03066b&order[followedCount]=desc"),
        fetchManga("limit=10&includedTags[]=423e2eae-a7a2-4a8b-ac03-a8351462d71d&order[followedCount]=desc"),
        fetchManga("limit=10&order[latestUploadedChapter]=desc")
      ]);

      setPopularManga(popular);
      setActionManga(action);
      setRomanceManga(romance);
      setRecentManga(recent);
    }

    loadHomePage();
  }, []);

  const fetchManga = async (params: string) => {
    try {
      const response = await fetch(
        `https://api.mangadex.org/manga?${params}&includes[]=cover_art`
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.data || [];
    } catch (error) {
      console.error("Error fetching manga:", error);
      return [];
    }
  };

  const handleSearch = async () => {
    console.log("Searching for:", searchQuery);
    if (!searchQuery) return;
    try {
      const response = await fetch(`https://api.mangadex.org/manga?title=${searchQuery}&limit=10&includes[]=cover_art`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
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

  const fetchChapters = async (mangaId: string) => {
    setLoadingChapters(true);
    try {
      const response = await fetch(
        `https://api.mangadex.org/manga/${mangaId}/feed?limit=20&order[chapter]=asc&translatedLanguage[]=en`
      );
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

  const renderMangaCard = (manga: any, width: number, height: number) => {
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
          marginHorizontal: 12
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
            style={{ width: width, height: height, borderRadius: 10 }}
          />
        </Pressable>
      </View>
    );
  };

  const onPressPagination = (index: number) => {
    ref.current?.scrollTo({
      count: index - progress.value,
      animated: true,
    });
  };
  
  return (
    <ScrollView contentContainerStyle={{ flexGrow: 1, alignItems: 'center', padding: 20 }}>
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
        style={{ width: width - 600, marginTop: 30, paddingHorizontal: (width - 900) / 2}}
        onProgressChange={(offsetProgress, absoluteProgress) => {
					progress.value = absoluteProgress;
				}}
        renderItem={({ item: manga }) => renderMangaCard(manga, 300, 450)}
      />

      <Pagination.Custom<{ color: string }>
				progress={progress}
				data={popularManga}
				size={10}
				dotStyle={{
					borderRadius: 16,
					backgroundColor: "#262626",
				}}
				activeDotStyle={{
					borderRadius: 8,
					width: 25,
					height: 15,
					overflow: "hidden",
					backgroundColor: "#b8b8b8",
				}}
				containerStyle={{
					gap: 8,
					marginBottom: 10,
					alignItems: "center",
					height: 10,
				}}
				horizontal
				onPress={onPressPagination}
				customReanimatedStyle={(progress, index, length) => {
					let val = Math.abs(progress - index);
					if (index === 0 && progress > length - 1) {
						val = Math.abs(progress - length);
					}
 
					return {
						transform: [
							{
								translateY: interpolate(
									val,
									[0, 1],
									[0, 0],
									Extrapolation.CLAMP,
								),
							},
						],
					};
				}}
			/>

      <View style={{marginTop: 30}}>
        <Text style={styles.category_header}>Recently Updated</Text>
        <FlatList
          horizontal={true}
          data={recentManga}
          style={styles.category_list}
          renderItem={({ item: manga }) => renderMangaCard(manga, 200, 300)}
        />
      </View>

      <View style={{marginTop: 30}}>
        <Text style={styles.category_header}>Action</Text>
        <FlatList
          horizontal={true}
          data={actionManga}
          style={styles.category_list}
          renderItem={({ item: manga }) => renderMangaCard(manga, 200, 300)}
        />
      </View>

      <View style={{marginTop: 30}}>
        <Text style={styles.category_header}>Romance</Text>
        <FlatList
          horizontal={true}
          data={romanceManga}
          style={styles.category_list}
          renderItem={({ item: manga }) => renderMangaCard(manga, 200, 300)}
        />
      </View>

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
  category_header: {
    fontSize: 21,
    marginBottom: 10
  },
  category_list: {
    marginLeft: -12,
    marginBottom: 25
  }
});