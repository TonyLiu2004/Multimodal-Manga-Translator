
export const GENRES = [
    { id: "", name: "All" }, 
    { id: "391b0423-d847-456f-aff0-8b0cfc03066b", name: "Action" },
    { id: "423e2eae-a7a2-4a8b-ac03-a8351462d71d", name: "Romance" },
    { id: "4d32cc48-9f00-4cca-9b5a-a839f0764984", name: "Comedy" },
    { id: "b9af3a63-f058-46de-a9a0-e0c13906197a", name: "Drama" },
    { id: "cdad7e68-1419-41dd-bdce-27753074a640", name: "Horror" },
    { id: "256c8bd9-4904-4360-bf4f-508a76d67183", name: "Sci-Fi" },
    { id: "e5301a23-ebd9-49dd-a0cb-2add944c7fe9", name: "Slice of Life" },
    { id: "69964a64-2f90-4d33-beeb-f3ed2875eb4c", name: "Sports" },
    { id: "eabc5b4c-6aff-42f3-b657-3e90cbd00b75", name: "Supernatural" },
    { id: "07251805-a27e-4d59-b488-f0bfbec15168", name: "Thriller" },
];

export const OTHER_TAGS = [
    { id: "followedCount_desc", name: "Most Popular" },
    { id: "latestUploadedChapter_desc", name: "Recently Updated" },
    { id: "rating_desc", name: "Highest Rated" },
    { id: "rating_asc", name: "Lowest Rated" },
    { id: "createdAt_desc", name: "Newest" },
];

export const SORT_MAP: Record<string, { order_by: string; order_direction: string }> = {
    followedCount_desc: { order_by: "followedCount", order_direction: "desc" },
    latestUploadedChapter_desc: { order_by: "latestUploadedChapter", order_direction: "desc" },
    rating_desc: { order_by: "rating", order_direction: "desc" },
    rating_asc: { order_by: "rating", order_direction: "asc" },
    createdAt_desc: { order_by: "createdAt", order_direction: "desc" },
};

export const UPDATE_STATUS = [
    {id: "ongoing", name: "Ongoing"},
    {id: "completed", name: "Completed"},
    {id: "hiatus", name: "Hiatus"},
    {id: "cancelled", name: "Cancelled"},
]
