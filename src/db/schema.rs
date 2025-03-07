// @generated automatically by Diesel CLI.

pub mod sql_types {
    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "bcd_event_type"))]
    pub struct BcdEventType;

    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "engine_type"))]
    pub struct EngineType;

    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "fxa_event_status_type"))]
    pub struct FxaEventStatusType;

    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "fxa_event_type"))]
    pub struct FxaEventType;

    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "locale"))]
    pub struct Locale;

    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "notification_type"))]
    pub struct NotificationType;

    #[derive(diesel::sql_types::SqlType)]
    #[diesel(postgres_type(name = "subscription_type"))]
    pub struct SubscriptionType;
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    activity_pings (id) {
        id -> Int8,
        user_id -> Int8,
        ping_at -> Timestamp,
        activity -> Jsonb,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    bcd_features (id) {
        id -> Int8,
        deprecated -> Nullable<Bool>,
        experimental -> Nullable<Bool>,
        mdn_url -> Nullable<Text>,
        path -> Text,
        short_title -> Nullable<Text>,
        source_file -> Text,
        spec_url -> Nullable<Text>,
        standard_track -> Nullable<Bool>,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;
    use super::sql_types::BcdEventType;
    use super::sql_types::EngineType;

    bcd_updates (id) {
        id -> Int8,
        browser_release -> Int8,
        created_at -> Timestamp,
        description -> Nullable<Text>,
        event_type -> BcdEventType,
        feature -> Int8,
        engines -> Array<Nullable<EngineType>>,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    browser_releases (id) {
        id -> Int8,
        browser -> Text,
        engine -> Text,
        engine_version -> Text,
        release_id -> Text,
        release_date -> Date,
        release_notes -> Nullable<Text>,
        status -> Nullable<Text>,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    browsers (name) {
        name -> Text,
        display_name -> Text,
        accepts_flags -> Nullable<Bool>,
        accepts_webextensions -> Nullable<Bool>,
        pref_url -> Nullable<Text>,
        preview_name -> Nullable<Text>,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    collection_items (id) {
        id -> Int8,
        created_at -> Timestamp,
        updated_at -> Timestamp,
        deleted_at -> Nullable<Timestamp>,
        document_id -> Int8,
        notes -> Nullable<Text>,
        custom_name -> Nullable<Text>,
        user_id -> Int8,
        multiple_collection_id -> Int8,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    documents (id) {
        id -> Int8,
        created_at -> Timestamp,
        updated_at -> Timestamp,
        absolute_uri -> Text,
        uri -> Text,
        metadata -> Nullable<Jsonb>,
        title -> Text,
        paths -> Array<Nullable<Text>>,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    multiple_collections (id) {
        id -> Int8,
        created_at -> Timestamp,
        updated_at -> Timestamp,
        deleted_at -> Nullable<Timestamp>,
        user_id -> Int8,
        notes -> Nullable<Text>,
        name -> Text,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;
    use super::sql_types::NotificationType;

    notification_data (id) {
        id -> Int8,
        created_at -> Timestamp,
        updated_at -> Timestamp,
        text -> Text,
        url -> Text,
        data -> Nullable<Jsonb>,
        title -> Text,
        #[sql_name = "type"]
        type_ -> NotificationType,
        document_id -> Int8,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    notifications (id) {
        id -> Int8,
        user_id -> Int8,
        starred -> Bool,
        read -> Bool,
        deleted_at -> Nullable<Timestamp>,
        notification_data_id -> Int8,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    raw_webhook_events_tokens (id) {
        id -> Int8,
        received_at -> Timestamp,
        token -> Text,
        error -> Text,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;
    use super::sql_types::Locale;

    settings (id) {
        id -> Int8,
        user_id -> Int8,
        locale_override -> Nullable<Locale>,
        mdnplus_newsletter -> Bool,
        no_ads -> Bool,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;
    use super::sql_types::SubscriptionType;

    users (id) {
        id -> Int8,
        created_at -> Timestamp,
        updated_at -> Timestamp,
        email -> Text,
        fxa_uid -> Varchar,
        fxa_refresh_token -> Varchar,
        avatar_url -> Nullable<Text>,
        subscription_type -> Nullable<SubscriptionType>,
        enforce_plus -> Nullable<SubscriptionType>,
        is_admin -> Bool,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;

    watched_items (user_id, document_id) {
        user_id -> Int8,
        document_id -> Int8,
        created_at -> Timestamp,
    }
}

diesel::table! {
    use diesel::sql_types::*;
    use crate::db::types::*;
    use super::sql_types::FxaEventType;
    use super::sql_types::FxaEventStatusType;

    webhook_events (id) {
        id -> Int8,
        fxa_uid -> Varchar,
        change_time -> Nullable<Timestamp>,
        issue_time -> Timestamp,
        typ -> FxaEventType,
        status -> FxaEventStatusType,
        payload -> Jsonb,
    }
}

diesel::joinable!(activity_pings -> users (user_id));
diesel::joinable!(bcd_updates -> bcd_features (feature));
diesel::joinable!(bcd_updates -> browser_releases (browser_release));
diesel::joinable!(browser_releases -> browsers (browser));
diesel::joinable!(collection_items -> documents (document_id));
diesel::joinable!(collection_items -> multiple_collections (multiple_collection_id));
diesel::joinable!(collection_items -> users (user_id));
diesel::joinable!(multiple_collections -> users (user_id));
diesel::joinable!(notification_data -> documents (document_id));
diesel::joinable!(notifications -> notification_data (notification_data_id));
diesel::joinable!(notifications -> users (user_id));
diesel::joinable!(settings -> users (user_id));
diesel::joinable!(watched_items -> documents (document_id));
diesel::joinable!(watched_items -> users (user_id));

diesel::allow_tables_to_appear_in_same_query!(
    activity_pings,
    bcd_features,
    bcd_updates,
    browser_releases,
    browsers,
    collection_items,
    documents,
    multiple_collections,
    notification_data,
    notifications,
    raw_webhook_events_tokens,
    settings,
    users,
    watched_items,
    webhook_events,
);
