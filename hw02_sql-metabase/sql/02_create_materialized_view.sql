
DROP MATERIALIZED VIEW IF EXISTS "student_hamed_hamzeh".mv_airbnb_neighbourhood_summary;

CREATE MATERIALIZED VIEW "student_hamed_hamzeh".mv_airbnb_neighbourhood_summary AS

WITH calendar_30 AS (
    SELECT
        listing_id,
        avg(price) as avg_calendar_price_30,
        avg(available::int) as availability_30_rate
    FROM core.calendar_day
    WHERE date BETWEEN
        (select min(date) from core.calendar_day)
        and (select min(date) from core.calendar_day) + interval '30 days'
    GROUP BY listing_id
),

review_counts AS (
    SELECT
        listing_id,
        count(*) as total_reviews
    FROM core.review
    GROUP BY listing_id
),

listing_base as (
    SELECT
        listing_id,
        neighbourhood_id,
        listing_price,
        minimum_nights
    FROM core.listing
),

features as (
    SELECT
        l.listing_id,
        l.neighbourhood_id,
        l.listing_price,
        l.minimum_nights,
        c.avg_calendar_price_30,
        c.availability_30_rate,
        r.total_reviews
    FROM listing_base l
    LEFT JOIN calendar_30 c ON l.listing_id = c.listing_id
    LEFT JOIN review_counts r ON l.listing_id = r.listing_id
),

with_neighbourhood as (
    SELECT
        f.*,
        n.name as neighbourhood
    FROM features f
    LEFT JOIN core.neighbourhood n
        ON f.neighbourhood_id = n.neighbourhood_id
)

SELECT
    neighbourhood,
    count(*) as num_listings,
    avg(listing_price) as avg_price,
    percentile_cont(0.5) within group (order by listing_price) as median_price,
    sum(total_reviews) as total_reviews,
    sum(total_reviews)::float / count(*) as reviews_per_listing,
    avg(availability_30_rate) as availability_30_rate
FROM with_neighbourhood
GROUP BY neighbourhood;
;

CREATE INDEX idx_mv_airbnb_neighbourhood
ON "student_hamed_hamzeh".mv_airbnb_neighbourhood_summary (neighbourhood);

CREATE INDEX idx_mv_airbnb_num_listings
ON "student_hamed_hamzeh".mv_airbnb_neighbourhood_summary (num_listings);
