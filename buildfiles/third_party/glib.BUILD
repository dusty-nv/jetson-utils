"""
Copyright 2021, NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

cc_binary(
    name = "libgobject-2.0.so",
    srcs = ["gobject_stub"],
    linkopts = ["-Wl,-soname,libgobject-2.0.so.0"],
    linkshared = True,
    visibility = ["//visibility:public"],
)

cc_import(
    name = "libgobject",
    shared_library = "libgobject-2.0.so",
)

# cc_library(
#     name = "glib",
#     hdrs = glob([
#         "**/*.h",
#     ]),
#     includes = [
#         ".",
#         "glib",
#         "_build/glib",
#         "_build",
#     ],
#     visibility = ["//visibility:public"],
#     deps = [
#         "libgobject",
#     ],
# )

config_setting(
    name = "aarch64-linux-gnu",
    define_values = {"multiarch": "aarch64-linux-gnu"},
)

config_setting(
    name = "x86_64-linux-gnu",
    define_values = {"multiarch": "x86_64-linux-gnu"},
)

cc_library(
    name = "glib",
    hdrs = glob([
        "external/include/glib-2.0/**/*.h",
        "external/lib/x86_64-linux-gnu/glib-2.0/include/**/*.h",
    ]) + select({
        ":aarch64-linux-gnu": ["external/lib/aarch64-linux-gnu/glib-2.0/include/**/*.h"],
        ":x86_64-linux-gnu": ["external/lib/x86_64-linux-gnu/glib-2.0/include/**/*.h"],
        "//conditions:default": [],
    }),
    copts = [
        "-Iexternal/include/glib-2.0",
        "-Iexternal/lib/x86_64-linux-gnu/glib-2.0/include",
    ] + select({
        ":aarch64-linux-gnu": ["-Iexternal/lib/aarch64-linux-gnu/glib-2.0/include"],
        ":x86_64-linux-gnu": ["-Iexternal/lib/x86_64-linux-gnu/glib-2.0/include"],
        "//conditions:default": [],
    }),
    includes = [
        "external/include/glib-2.0",
        "external/lib/x86_64-linux-gnu/glib-2.0/include",
    ] + select({
        ":aarch64-linux-gnu": ["external/lib/aarch64-linux-gnu/glib-2.0/include"],
        ":x86_64-linux-gnu": ["external/lib/x86_64-linux-gnu/glib-2.0/include"],
        "//conditions:default": [],
    }),
    linkopts = [
        "-Lexternal/lib/x86_64-linux-gnu",
        "-lglib-2.0",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "libgobject",
    ],
)

gobject_stub_content = """
extern \\"C\\" {
void g_array_get_type() {}
void g_binding_flags_get_type() {}
void g_binding_get_flags() {}
void g_binding_get_source() {}
void g_binding_get_source_property() {}
void g_binding_get_target() {}
void g_binding_get_target_property() {}
void g_binding_get_type() {}
void g_binding_unbind() {}
void g_boxed_copy() {}
void g_boxed_free() {}
void g_boxed_type_register_static() {}
void g_byte_array_get_type() {}
void g_bytes_get_type() {}
void g_cclosure_marshal_BOOLEAN__BOXED_BOXED() {}
void g_cclosure_marshal_BOOLEAN__BOXED_BOXEDv() {}
void g_cclosure_marshal_BOOLEAN__FLAGS() {}
void g_cclosure_marshal_BOOLEAN__FLAGSv() {}
void g_cclosure_marshal_generic() {}
void g_cclosure_marshal_generic_va() {}
void g_cclosure_marshal_STRING__OBJECT_POINTER() {}
void g_cclosure_marshal_STRING__OBJECT_POINTERv() {}
void g_cclosure_marshal_VOID__BOOLEAN() {}
void g_cclosure_marshal_VOID__BOOLEANv() {}
void g_cclosure_marshal_VOID__BOXED() {}
void g_cclosure_marshal_VOID__BOXEDv() {}
void g_cclosure_marshal_VOID__CHAR() {}
void g_cclosure_marshal_VOID__CHARv() {}
void g_cclosure_marshal_VOID__DOUBLE() {}
void g_cclosure_marshal_VOID__DOUBLEv() {}
void g_cclosure_marshal_VOID__ENUM() {}
void g_cclosure_marshal_VOID__ENUMv() {}
void g_cclosure_marshal_VOID__FLAGS() {}
void g_cclosure_marshal_VOID__FLAGSv() {}
void g_cclosure_marshal_VOID__FLOAT() {}
void g_cclosure_marshal_VOID__FLOATv() {}
void g_cclosure_marshal_VOID__INT() {}
void g_cclosure_marshal_VOID__INTv() {}
void g_cclosure_marshal_VOID__LONG() {}
void g_cclosure_marshal_VOID__LONGv() {}
void g_cclosure_marshal_VOID__OBJECT() {}
void g_cclosure_marshal_VOID__OBJECTv() {}
void g_cclosure_marshal_VOID__PARAM() {}
void g_cclosure_marshal_VOID__PARAMv() {}
void g_cclosure_marshal_VOID__POINTER() {}
void g_cclosure_marshal_VOID__POINTERv() {}
void g_cclosure_marshal_VOID__STRING() {}
void g_cclosure_marshal_VOID__STRINGv() {}
void g_cclosure_marshal_VOID__UCHAR() {}
void g_cclosure_marshal_VOID__UCHARv() {}
void g_cclosure_marshal_VOID__UINT() {}
void g_cclosure_marshal_VOID__UINT_POINTER() {}
void g_cclosure_marshal_VOID__UINT_POINTERv() {}
void g_cclosure_marshal_VOID__UINTv() {}
void g_cclosure_marshal_VOID__ULONG() {}
void g_cclosure_marshal_VOID__ULONGv() {}
void g_cclosure_marshal_VOID__VARIANT() {}
void g_cclosure_marshal_VOID__VARIANTv() {}
void g_cclosure_marshal_VOID__VOID() {}
void g_cclosure_marshal_VOID__VOIDv() {}
void g_cclosure_new() {}
void g_cclosure_new_object() {}
void g_cclosure_new_object_swap() {}
void g_cclosure_new_swap() {}
void g_checksum_get_type() {}
void g_clear_object() {}
void g_clear_signal_handler() {}
void g_closure_add_finalize_notifier() {}
void g_closure_add_invalidate_notifier() {}
void g_closure_add_marshal_guards() {}
void g_closure_get_type() {}
void g_closure_invalidate() {}
void g_closure_invoke() {}
void g_closure_new_object() {}
void g_closure_new_simple() {}
void g_closure_ref() {}
void g_closure_remove_finalize_notifier() {}
void g_closure_remove_invalidate_notifier() {}
void g_closure_set_marshal() {}
void g_closure_set_meta_marshal() {}
void g_closure_sink() {}
void g_closure_unref() {}
void g_date_get_type() {}
void g_date_time_get_type() {}
void g_enum_complete_type_info() {}
void g_enum_get_value() {}
void g_enum_get_value_by_name() {}
void g_enum_get_value_by_nick() {}
void g_enum_register_static() {}
void g_enum_to_string() {}
void g_error_get_type() {}
void g_flags_complete_type_info() {}
void g_flags_get_first_value() {}
void g_flags_get_value_by_name() {}
void g_flags_get_value_by_nick() {}
void g_flags_register_static() {}
void g_flags_to_string() {}
void g_gstring_get_type() {}
void g_gtype_get_type() {}
void g_hash_table_get_type() {}
void g_initially_unowned_get_type() {}
void g_io_channel_get_type() {}
void g_io_condition_get_type() {}
void g_key_file_get_type() {}
void g_main_context_get_type() {}
void g_main_loop_get_type() {}
void g_mapped_file_get_type() {}
void g_markup_parse_context_get_type() {}
void g_match_info_get_type() {}
void g_normalize_mode_get_type() {}
void g_object_add_toggle_ref() {}
void g_object_add_weak_pointer() {}
void g_object_bind_property() {}
void g_object_bind_property_full() {}
void g_object_bind_property_with_closures() {}
void g_object_class_find_property() {}
void g_object_class_install_properties() {}
void g_object_class_install_property() {}
void g_object_class_list_properties() {}
void g_object_class_override_property() {}
void g_object_compat_control() {}
void g_object_connect() {}
void g_object_disconnect() {}
void g_object_dup_data() {}
void g_object_dup_qdata() {}
void g_object_force_floating() {}
void g_object_freeze_notify() {}
void g_object_get() {}
void g_object_get_data() {}
void g_object_get_property() {}
void g_object_get_qdata() {}
void g_object_get_type() {}
void g_object_getv() {}
void g_object_get_valist() {}
void g_object_interface_find_property() {}
void g_object_interface_install_property() {}
void g_object_interface_list_properties() {}
void g_object_is_floating() {}
void g_object_new() {}
void g_object_newv() {}
void g_object_new_valist() {}
void g_object_new_with_properties() {}
void g_object_notify() {}
void g_object_notify_by_pspec() {}
void g_object_ref() {}
void g_object_ref_sink() {}
void g_object_remove_toggle_ref() {}
void g_object_remove_weak_pointer() {}
void g_object_replace_data() {}
void g_object_replace_qdata() {}
void g_object_run_dispose() {}
void g_object_set() {}
void g_object_set_data() {}
void g_object_set_data_full() {}
void g_object_set_property() {}
void g_object_set_qdata() {}
void g_object_set_qdata_full() {}
void g_object_setv() {}
void g_object_set_valist() {}
void g_object_steal_data() {}
void g_object_steal_qdata() {}
void g_object_thaw_notify() {}
void g_object_unref() {}
void g_object_watch_closure() {}
void g_object_weak_ref() {}
void g_object_weak_unref() {}
void g_option_group_get_type() {}
void g_param_spec_boolean() {}
void g_param_spec_boxed() {}
void g_param_spec_char() {}
void g_param_spec_double() {}
void g_param_spec_enum() {}
void g_param_spec_flags() {}
void g_param_spec_float() {}
void g_param_spec_get_blurb() {}
void g_param_spec_get_default_value() {}
void g_param_spec_get_name() {}
void g_param_spec_get_name_quark() {}
void g_param_spec_get_nick() {}
void g_param_spec_get_qdata() {}
void g_param_spec_get_redirect_target() {}
void g_param_spec_gtype() {}
void g_param_spec_int() {}
void g_param_spec_int64() {}
void g_param_spec_internal() {}
void g_param_spec_long() {}
void g_param_spec_object() {}
void g_param_spec_override() {}
void g_param_spec_param() {}
void g_param_spec_pointer() {}
void g_param_spec_pool_insert() {}
void g_param_spec_pool_list() {}
void g_param_spec_pool_list_owned() {}
void g_param_spec_pool_lookup() {}
void g_param_spec_pool_new() {}
void g_param_spec_pool_remove() {}
void g_param_spec_ref() {}
void g_param_spec_ref_sink() {}
void g_param_spec_set_qdata() {}
void g_param_spec_set_qdata_full() {}
void g_param_spec_sink() {}
void g_param_spec_steal_qdata() {}
void g_param_spec_string() {}
void g_param_spec_types() {}
void g_param_spec_uchar() {}
void g_param_spec_uint() {}
void g_param_spec_uint64() {}
void g_param_spec_ulong() {}
void g_param_spec_unichar() {}
void g_param_spec_unref() {}
void g_param_spec_value_array() {}
void g_param_spec_variant() {}
void g_param_type_register_static() {}
void g_param_value_convert() {}
void g_param_value_defaults() {}
void g_param_values_cmp() {}
void g_param_value_set_default() {}
void g_param_value_validate() {}
void g_pointer_type_register_static() {}
void g_pollfd_get_type() {}
void g_ptr_array_get_type() {}
void g_regex_get_type() {}
void g_signal_accumulator_first_wins() {}
void g_signal_accumulator_true_handled() {}
void g_signal_add_emission_hook() {}
void g_signal_chain_from_overridden() {}
void g_signal_chain_from_overridden_handler() {}
void g_signal_connect_closure() {}
void g_signal_connect_closure_by_id() {}
void g_signal_connect_data() {}
void g_signal_connect_object() {}
void g_signal_emit() {}
void g_signal_emit_by_name() {}
void g_signal_emitv() {}
void g_signal_emit_valist() {}
void g_signal_get_invocation_hint() {}
void g_signal_handler_block() {}
void g_signal_handler_disconnect() {}
void g_signal_handler_find() {}
void g_signal_handler_is_connected() {}
void g_signal_handlers_block_matched() {}
void g_signal_handlers_destroy() {}
void g_signal_handlers_disconnect_matched() {}
void g_signal_handlers_unblock_matched() {}
void g_signal_handler_unblock() {}
void g_signal_has_handler_pending() {}
void g_signal_list_ids() {}
void g_signal_lookup() {}
void g_signal_name() {}
void g_signal_new() {}
void g_signal_new_class_handler() {}
void g_signal_newv() {}
void g_signal_new_valist() {}
void g_signal_override_class_closure() {}
void g_signal_override_class_handler() {}
void g_signal_parse_name() {}
void g_signal_query() {}
void g_signal_remove_emission_hook() {}
void g_signal_set_va_marshaller() {}
void g_signal_stop_emission() {}
void g_signal_stop_emission_by_name() {}
void g_signal_type_cclosure_new() {}
void g_source_get_type() {}
void g_source_set_closure() {}
void g_source_set_dummy_callback() {}
void g_strdup_value_contents() {}
void g_strv_get_type() {}
void g_thread_get_type() {}
void g_time_zone_get_type() {}
void g_type_add_class_cache_func() {}
void g_type_add_class_private() {}
void g_type_add_instance_private() {}
void g_type_add_interface_check() {}
void g_type_add_interface_dynamic() {}
void g_type_add_interface_static() {}
void g_type_check_class_cast() {}
void g_type_check_class_is_a() {}
void g_type_check_instance() {}
void g_type_check_instance_cast() {}
void g_type_check_instance_is_a() {}
void g_type_check_instance_is_fundamentally_a() {}
void g_type_check_is_value_type() {}
void g_type_check_value() {}
void g_type_check_value_holds() {}
void g_type_children() {}
void g_type_class_add_private() {}
void g_type_class_adjust_private_offset() {}
void g_type_class_get_instance_private_offset() {}
void g_type_class_get_private() {}
void g_type_class_peek() {}
void g_type_class_peek_parent() {}
void g_type_class_peek_static() {}
void g_type_class_ref() {}
void g_type_class_unref() {}
void g_type_class_unref_uncached() {}
void g_type_create_instance() {}
void g_type_default_interface_peek() {}
void g_type_default_interface_ref() {}
void g_type_default_interface_unref() {}
void g_type_depth() {}
void g_type_ensure() {}
void g_type_free_instance() {}
void g_type_from_name() {}
void g_type_fundamental() {}
void g_type_fundamental_next() {}
void g_type_get_instance_count() {}
void g_type_get_plugin() {}
void g_type_get_qdata() {}
void g_type_get_type_registration_serial() {}
void g_type_init() {}
void g_type_init_with_debug_flags() {}
void g_type_instance_get_private() {}
void g_type_interface_add_prerequisite() {}
void g_type_interface_get_plugin() {}
void g_type_interface_peek() {}
void g_type_interface_peek_parent() {}
void g_type_interface_prerequisites() {}
void g_type_interfaces() {}
void g_type_is_a() {}
void g_type_module_add_interface() {}
void g_type_module_get_type() {}
void g_type_module_register_enum() {}
void g_type_module_register_flags() {}
void g_type_module_register_type() {}
void g_type_module_set_name() {}
void g_type_module_unuse() {}
void g_type_module_use() {}
void g_type_name() {}
void g_type_name_from_class() {}
void g_type_name_from_instance() {}
void g_type_next_base() {}
void g_type_parent() {}
void g_type_plugin_complete_interface_info() {}
void g_type_plugin_complete_type_info() {}
void g_type_plugin_get_type() {}
void g_type_plugin_unuse() {}
void g_type_plugin_use() {}
void g_type_qname() {}
void g_type_query() {}
void g_type_register_dynamic() {}
void g_type_register_fundamental() {}
void g_type_register_static() {}
void g_type_register_static_simple() {}
void g_type_remove_class_cache_func() {}
void g_type_remove_interface_check() {}
void g_type_set_qdata() {}
void g_type_test_flags() {}
void g_type_value_table_peek() {}
void g_unicode_break_type_get_type() {}
void g_unicode_script_get_type() {}
void g_unicode_type_get_type() {}
void g_value_array_append() {}
void g_value_array_copy() {}
void g_value_array_free() {}
void g_value_array_get_nth() {}
void g_value_array_get_type() {}
void g_value_array_insert() {}
void g_value_array_new() {}
void g_value_array_prepend() {}
void g_value_array_remove() {}
void g_value_array_sort() {}
void g_value_array_sort_with_data() {}
void g_value_copy() {}
void g_value_dup_boxed() {}
void g_value_dup_object() {}
void g_value_dup_param() {}
void g_value_dup_string() {}
void g_value_dup_variant() {}
void g_value_fits_pointer() {}
void g_value_get_boolean() {}
void g_value_get_boxed() {}
void g_value_get_char() {}
void g_value_get_double() {}
void g_value_get_enum() {}
void g_value_get_flags() {}
void g_value_get_float() {}
void g_value_get_gtype() {}
void g_value_get_int() {}
void g_value_get_int64() {}
void g_value_get_long() {}
void g_value_get_object() {}
void g_value_get_param() {}
void g_value_get_pointer() {}
void g_value_get_schar() {}
void g_value_get_string() {}
void g_value_get_type() {}
void g_value_get_uchar() {}
void g_value_get_uint() {}
void g_value_get_uint64() {}
void g_value_get_ulong() {}
void g_value_get_variant() {}
void g_value_init() {}
void g_value_init_from_instance() {}
void g_value_peek_pointer() {}
void g_value_register_transform_func() {}
void g_value_reset() {}
void g_value_set_boolean() {}
void g_value_set_boxed() {}
void g_value_set_boxed_take_ownership() {}
void g_value_set_char() {}
void g_value_set_double() {}
void g_value_set_enum() {}
void g_value_set_flags() {}
void g_value_set_float() {}
void g_value_set_gtype() {}
void g_value_set_instance() {}
void g_value_set_int() {}
void g_value_set_int64() {}
void g_value_set_long() {}
void g_value_set_object() {}
void g_value_set_object_take_ownership() {}
void g_value_set_param() {}
void g_value_set_param_take_ownership() {}
void g_value_set_pointer() {}
void g_value_set_schar() {}
void g_value_set_static_boxed() {}
void g_value_set_static_string() {}
void g_value_set_string() {}
void g_value_set_string_take_ownership() {}
void g_value_set_uchar() {}
void g_value_set_uint() {}
void g_value_set_uint64() {}
void g_value_set_ulong() {}
void g_value_set_variant() {}
void g_value_take_boxed() {}
void g_value_take_object() {}
void g_value_take_param() {}
void g_value_take_string() {}
void g_value_take_variant() {}
void g_value_transform() {}
void g_value_type_compatible() {}
void g_value_type_transformable() {}
void g_value_unset() {}
void g_variant_builder_get_type() {}
void g_variant_dict_get_type() {}
void g_variant_get_gtype() {}
void g_variant_type_get_gtype() {}
void g_weak_ref_clear() {}
void g_weak_ref_get() {}
void g_weak_ref_init() {}
void g_weak_ref_set() {}
}
"""

genrule(
    name = "gobject_stub",
    outs = ["gobject_stub.cpp"],
    cmd = "echo \"" + gobject_stub_content + "\" > $@",
)
