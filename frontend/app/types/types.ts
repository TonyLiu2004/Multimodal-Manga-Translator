export interface Manga {
  id: string;
  attributes: {
    title: { [key: string]: string };
    description: { en?: string };
  };
  relationships: {
    type: string;
    attributes?: { fileName?: string };
  }[];
}

export interface Chapter {
  id: string;
  chapter: string;
  title: string;
  pages: number;
}