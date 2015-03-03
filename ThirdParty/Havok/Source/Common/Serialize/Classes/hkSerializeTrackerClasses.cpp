/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Serialize/hkSerialize.h>
static const char s_libraryName[] = "hkSerialize";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkSerializeRegister() {}

#include <Common/Serialize/Copier/hkObjectCopier.h>


// hkObjectCopier ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectCopier)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ObjectCopierFlagBits)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectCopier)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkObjectCopier, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkObjectCopier, ObjectCopierFlagBits, s_libraryName)

#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>


// hkDataWorldDict ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataWorldDict)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataWorldDict)
    HK_TRACKER_MEMBER(hkDataWorldDict, m_tracker, 0, "hkDataWorldDict::ObjectTracker*") // class hkDataWorldDict::ObjectTracker*
    HK_TRACKER_MEMBER(hkDataWorldDict, m_allocator, 0, "hkMemoryAllocator*") // class hkMemoryAllocator*
    HK_TRACKER_MEMBER(hkDataWorldDict, m_typeManager, 0, "hkTypeManager") // class hkTypeManager
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDataWorldDict, s_libraryName, hkDataWorld)

#include <Common/Serialize/Data/Native/hkDataObjectNative.h>


// hkDataWorldNative ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataWorldNative)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataWorldNative)
    HK_TRACKER_MEMBER(hkDataWorldNative, m_reg, 0, "hkClassNameRegistry *") // const class hkClassNameRegistry *
    HK_TRACKER_MEMBER(hkDataWorldNative, m_vtable, 0, "hkVtableClassRegistry *") // const class hkVtableClassRegistry *
    HK_TRACKER_MEMBER(hkDataWorldNative, m_infoReg, 0, "hkTypeInfoRegistry *") // const class hkTypeInfoRegistry *
    HK_TRACKER_MEMBER(hkDataWorldNative, m_classes, 0, "hkStringMap<hkDataClassNative*, hkContainerHeapAllocator>") // class hkStringMap< class hkDataClassNative*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDataWorldNative, m_contents, 0, "hkVariant") // hkVariant
    HK_TRACKER_MEMBER(hkDataWorldNative, m_typeManager, 0, "hkTypeManager") // class hkTypeManager
    HK_TRACKER_MEMBER(hkDataWorldNative, m_buffer, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDataWorldNative, s_libraryName, hkDataWorld)

#include <Common/Serialize/Data/Util/hkDataObjectToNative.h>


// hkDataObjectToNative ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObjectToNative)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PointerInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Alloc)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DummyArray)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DummyRelArray)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CopyInfoOut)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObjectToNative)
    HK_TRACKER_MEMBER(hkDataObjectToNative, m_classReg, 0, "hkClassNameRegistry*") // const class hkClassNameRegistry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObjectToNative, s_libraryName)


// PointerInfo hkDataObjectToNative

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObjectToNative::PointerInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObjectToNative::PointerInfo)
    HK_TRACKER_MEMBER(hkDataObjectToNative::PointerInfo, m_handle, 0, "hkDataObject_Handle") // struct hkDataObject_Handle
    HK_TRACKER_MEMBER(hkDataObjectToNative::PointerInfo, m_addr, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObjectToNative::PointerInfo, s_libraryName)


// Alloc hkDataObjectToNative

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObjectToNative::Alloc)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObjectToNative::Alloc)
    HK_TRACKER_MEMBER(hkDataObjectToNative::Alloc, m_addr, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObjectToNative::Alloc, s_libraryName)


// DummyArray hkDataObjectToNative

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObjectToNative::DummyArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObjectToNative::DummyArray)
    HK_TRACKER_MEMBER(hkDataObjectToNative::DummyArray, data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObjectToNative::DummyArray, s_libraryName)


// DummyRelArray hkDataObjectToNative
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDataObjectToNative, DummyRelArray, s_libraryName)


// CopyInfoOut hkDataObjectToNative

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObjectToNative::CopyInfoOut)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObjectToNative::CopyInfoOut)
    HK_TRACKER_MEMBER(hkDataObjectToNative::CopyInfoOut, pointersOut, 0, "hkArray<hkDataObjectToNative::PointerInfo, hkContainerTempAllocator>") // hkArray< struct hkDataObjectToNative::PointerInfo, struct hkContainerTempAllocator >
    HK_TRACKER_MEMBER(hkDataObjectToNative::CopyInfoOut, allocs, 0, "hkArray<hkDataObjectToNative::Alloc, hkContainerTempAllocator>") // hkArray< struct hkDataObjectToNative::Alloc, struct hkContainerTempAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObjectToNative::CopyInfoOut, s_libraryName)

#include <Common/Serialize/Data/Util/hkDataWorldCloner.h>


// hkDataWorldCloner ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataWorldCloner)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataWorldCloner)
    HK_TRACKER_MEMBER(hkDataWorldCloner, m_copied, 0, "hkMap<hkDataObject_Handle, hkDataObject_Handle, hkMapOperations<hkDataObject_Handle>, hkContainerHeapAllocator>") // class hkMap< struct hkDataObject_Handle, struct hkDataObject_Handle, struct hkMapOperations< struct hkDataObject_Handle >, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataWorldCloner, s_libraryName)

#include <Common/Serialize/Data/hkDataObject.h>


// hkDataObject_Value ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObject_Value)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObject_Value)
    HK_TRACKER_MEMBER(hkDataObject_Value, m_impl, 0, "hkDataObjectImpl*") // class hkDataObjectImpl*
    HK_TRACKER_MEMBER(hkDataObject_Value, m_handle, 0, "_hkDataObject_MemberHandle*") // const struct _hkDataObject_MemberHandle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObject_Value, s_libraryName)


// hkDataObject ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObject)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObject)
    HK_TRACKER_MEMBER(hkDataObject, m_impl, 0, "hkDataObjectImpl*") // class hkDataObjectImpl*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObject, s_libraryName)


// hkDataArray_Value ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataArray_Value)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataArray_Value)
    HK_TRACKER_MEMBER(hkDataArray_Value, m_impl, 0, "hkDataArrayImpl*") // class hkDataArrayImpl*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataArray_Value, s_libraryName)


// hkDataArray ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataArray)
    HK_TRACKER_MEMBER(hkDataArray, m_impl, 0, "hkDataArrayImpl*") // class hkDataArrayImpl*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataArray, s_libraryName)


// hkDataClass ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataClass)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cinfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataClass)
    HK_TRACKER_MEMBER(hkDataClass, m_impl, 0, "hkDataClassImpl*") // class hkDataClassImpl*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataClass, s_libraryName)


// Cinfo hkDataClass

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataClass::Cinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Member)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataClass::Cinfo)
    HK_TRACKER_MEMBER(hkDataClass::Cinfo, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkDataClass::Cinfo, parent, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkDataClass::Cinfo, members, 0, "hkArray<hkDataClass::Cinfo::Member, hkContainerHeapAllocator>") // hkArray< struct hkDataClass::Cinfo::Member, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataClass::Cinfo, s_libraryName)


// Member hkDataClass::Cinfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataClass::Cinfo::Member)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataClass::Cinfo::Member)
    HK_TRACKER_MEMBER(hkDataClass::Cinfo::Member, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkDataClass::Cinfo::Member, type, 0, "hkTypeManager::Type*") // struct hkTypeManager::Type*
    HK_TRACKER_MEMBER(hkDataClass::Cinfo::Member, valuePtr, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataClass::Cinfo::Member, s_libraryName)


// hkDataWorld ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataWorld)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DataWorldType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataWorld)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkDataWorld, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDataWorld, DataWorldType, s_libraryName)

 // Skipping Class hkMapOperations< struct hkDataObject_Handle > as it is a template

#include <Common/Serialize/Data/hkDataObjectDeclarations.h>


// hkDataClass_MemberInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataClass_MemberInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataClass_MemberInfo)
    HK_TRACKER_MEMBER(hkDataClass_MemberInfo, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkDataClass_MemberInfo, m_owner, 0, "hkDataClassImpl*") // const class hkDataClassImpl*
    HK_TRACKER_MEMBER(hkDataClass_MemberInfo, m_type, 0, "hkTypeManager::Type*") // struct hkTypeManager::Type*
    HK_TRACKER_MEMBER(hkDataClass_MemberInfo, m_valuePtr, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataClass_MemberInfo, s_libraryName)


// hkDataObject_Handle ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataObject_Handle)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataObject_Handle)
    HK_TRACKER_MEMBER(hkDataObject_Handle, p0, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkDataObject_Handle, p1, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataObject_Handle, s_libraryName)

#include <Common/Serialize/Data/hkDataObjectImpl.h>


// hkDataRefCounted ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDataRefCounted)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDataRefCounted)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkDataRefCounted, s_libraryName)

// hk.MemoryTracker ignore hkDataArrayImpl
// hk.MemoryTracker ignore hkDataClassImpl
// hk.MemoryTracker ignore hkDataObjectImpl
#include <Common/Serialize/Packfile/Binary/hkBinaryPackfileWriter.h>


// hkBinaryPackfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryPackfileWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryPackfileWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBinaryPackfileWriter, s_libraryName, hkPackfileWriter)

#include <Common/Serialize/Packfile/Binary/hkPackfileHeader.h>


// hkPackfileHeader ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPackfileHeader, s_libraryName)

#include <Common/Serialize/Packfile/Binary/hkPackfileSectionHeader.h>


// hkPackfileSectionHeader ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkPackfileSectionHeader, s_libraryName)

#include <Common/Serialize/Packfile/hkPackfileData.h>


// hkPackfileData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Chunk)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileData)
    HK_TRACKER_MEMBER(hkPackfileData, m_topLevelObject, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkPackfileData, m_name, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkPackfileData, m_trackedObjects, 0, "hkPointerMap<void*, char*, hkContainerHeapAllocator>") // class hkPointerMap< void*, const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileData, m_trackedTypes, 0, "hkStringMap<hkTypeInfo*, hkContainerHeapAllocator>") // class hkStringMap< const class hkTypeInfo*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileData, m_packfileClassRegistry, 0, "hkClassNameRegistry *") // const class hkClassNameRegistry *
    HK_TRACKER_MEMBER(hkPackfileData, m_memory, 0, "hkArray<void*, hkContainerHeapAllocator>") // hkArray< void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileData, m_chunks, 0, "hkArray<hkPackfileData::Chunk, hkContainerHeapAllocator>") // hkArray< struct hkPackfileData::Chunk, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileData, m_exports, 0, "hkArray<hkResource::Export, hkContainerHeapAllocator>") // hkArray< struct hkResource::Export, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileData, m_imports, 0, "hkArray<hkResource::Import, hkContainerHeapAllocator>") // hkArray< struct hkResource::Import, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileData, m_postFinishObjects, 0, "hkArray<hkVariant, hkContainerHeapAllocator>") // hkArray< hkVariant, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkPackfileData, s_libraryName, hkResource)


// Chunk hkPackfileData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileData::Chunk)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileData::Chunk)
    HK_TRACKER_MEMBER(hkPackfileData::Chunk, pointer, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkPackfileData::Chunk, s_libraryName)

#include <Common/Serialize/Packfile/hkPackfileWriter.h>


// hkPackfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AddObjectListener)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Options)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PendingWrite)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileWriter)
    HK_TRACKER_MEMBER(hkPackfileWriter, m_pendingWrites, 0, "hkArray<hkPackfileWriter::PendingWrite, hkContainerHeapAllocator>") // hkArray< struct hkPackfileWriter::PendingWrite, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_knownObjects, 0, "hkPointerMap<void*, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< const void*, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_imports, 0, "hkPointerMap<void*, char*, hkContainerHeapAllocator>") // class hkPointerMap< const void*, const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_exports, 0, "hkPointerMap<void*, char*, hkContainerHeapAllocator>") // class hkPointerMap< const void*, const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_knownClasses, 0, "hkStringMap<hkClass*, hkContainerHeapAllocator>") // class hkStringMap< const hkClass*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_replacements, 0, "hkPointerMap<void*, void*, hkContainerHeapAllocator>") // class hkPointerMap< const void*, const void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_knownSections, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_sectionTagToIndex, 0, "hkStringMap<hkInt32, hkContainerHeapAllocator>") // class hkStringMap< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_objectsWithUnregisteredClass, 0, "hkArray<hkVariant, hkContainerHeapAllocator>") // hkArray< hkVariant, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_sectionOverrideByPointer, 0, "hkPointerMap<void*, hkUint32, hkContainerHeapAllocator>") // class hkPointerMap< const void*, hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_sectionOverrideByType, 0, "hkStringMap<hkUint32, hkContainerHeapAllocator>") // class hkStringMap< hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_pwIndexesFromReferencedPointer, 0, "hkSerializeMultiMap<void*, hkInt32, hkPointerMap<void*, hkInt32, hkContainerHeapAllocator> >") // class hkSerializeMultiMap< const void*, hkInt32, class hkPointerMap< const void*, hkInt32, struct hkContainerHeapAllocator > >
    HK_TRACKER_MEMBER(hkPackfileWriter, m_startOptions, 0, "hkPackfileWriter::Options") // struct hkPackfileWriter::Options
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkPackfileWriter, s_libraryName, hkReferencedObject)


// AddObjectListener hkPackfileWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileWriter::AddObjectListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileWriter::AddObjectListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkPackfileWriter::AddObjectListener, s_libraryName, hkReferencedObject)


// Options hkPackfileWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileWriter::Options)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileWriter::Options)
    HK_TRACKER_MEMBER(hkPackfileWriter::Options, m_contentsVersion, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkPackfileWriter::Options, s_libraryName)


// PendingWrite hkPackfileWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileWriter::PendingWrite)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileWriter::PendingWrite)
    HK_TRACKER_MEMBER(hkPackfileWriter::PendingWrite, m_pointer, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkPackfileWriter::PendingWrite, m_klass, 0, "hkClass*") // const hkClass*
    HK_TRACKER_MEMBER(hkPackfileWriter::PendingWrite, m_origPointer, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkPackfileWriter::PendingWrite, m_origClass, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkPackfileWriter::PendingWrite, s_libraryName)

#include <Common/Serialize/Resource/hkObjectResource.h>


// hkObjectResource ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectResource)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectResource)
    HK_TRACKER_MEMBER(hkObjectResource, m_topLevelObject, 0, "hkVariant") // hkVariant
    HK_TRACKER_MEMBER(hkObjectResource, m_classRegistry, 0, "hkClassNameRegistry *") // const class hkClassNameRegistry *
    HK_TRACKER_MEMBER(hkObjectResource, m_typeRegistry, 0, "hkTypeInfoRegistry *") // const class hkTypeInfoRegistry *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkObjectResource, s_libraryName, hkResource)

#include <Common/Serialize/Resource/hkResource.h>


// hkResource ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResource)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Export)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Import)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResource)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkResource, s_libraryName, hkReferencedObject)


// Export hkResource

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResource::Export)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResource::Export)
    HK_TRACKER_MEMBER(hkResource::Export, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkResource::Export, data, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkResource::Export, s_libraryName)


// Import hkResource

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResource::Import)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResource::Import)
    HK_TRACKER_MEMBER(hkResource::Import, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkResource::Import, location, 0, "void**") // void**
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkResource::Import, s_libraryName)

#include <Common/Serialize/ResourceDatabase/hkResourceHandle.h>


// hkResourceBase ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResourceBase)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResourceBase)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkResourceBase, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkResourceBase, Type, s_libraryName)


// hkResourceHandle ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResourceHandle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Link)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResourceHandle)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkResourceHandle, s_libraryName, hkResourceBase)


// Link hkResourceHandle

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResourceHandle::Link)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResourceHandle::Link)
    HK_TRACKER_MEMBER(hkResourceHandle::Link, m_memberName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkResourceHandle::Link, m_externalId, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkResourceHandle::Link, m_memberAccessor, 0, "hkClassMemberAccessor") // class hkClassMemberAccessor
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkResourceHandle::Link, s_libraryName)


// hkResourceContainer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResourceContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResourceContainer)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkResourceContainer, s_libraryName, hkResourceBase)


// hkResourceMap ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkResourceMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkResourceMap)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkResourceMap, s_libraryName)


// hkMemoryResourceHandle ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryResourceHandle)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ExternalLink)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryResourceHandle)
    HK_TRACKER_MEMBER(hkMemoryResourceHandle, m_variant, 0, "hkReferencedObject *") // class hkReferencedObject *
    HK_TRACKER_MEMBER(hkMemoryResourceHandle, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkMemoryResourceHandle, m_references, 0, "hkArray<hkMemoryResourceHandle::ExternalLink, hkContainerHeapAllocator>") // hkArray< struct hkMemoryResourceHandle::ExternalLink, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryResourceHandle, s_libraryName, hkResourceHandle)


// ExternalLink hkMemoryResourceHandle

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryResourceHandle::ExternalLink)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryResourceHandle::ExternalLink)
    HK_TRACKER_MEMBER(hkMemoryResourceHandle::ExternalLink, m_memberName, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkMemoryResourceHandle::ExternalLink, m_externalId, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkMemoryResourceHandle::ExternalLink, s_libraryName)


// hkMemoryResourceContainer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemoryResourceContainer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemoryResourceContainer)
    HK_TRACKER_MEMBER(hkMemoryResourceContainer, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkMemoryResourceContainer, m_parent, 0, "hkMemoryResourceContainer*") // class hkMemoryResourceContainer*
    HK_TRACKER_MEMBER(hkMemoryResourceContainer, m_resourceHandles, 0, "hkArray<hkMemoryResourceHandle *, hkContainerHeapAllocator>") // hkArray< class hkMemoryResourceHandle *, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkMemoryResourceContainer, m_children, 0, "hkArray<hkMemoryResourceContainer *, hkContainerHeapAllocator>") // hkArray< class hkMemoryResourceContainer *, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemoryResourceContainer, s_libraryName, hkResourceContainer)


// hkContainerResourceMap ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkContainerResourceMap)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkContainerResourceMap)
    HK_TRACKER_MEMBER(hkContainerResourceMap, m_resources, 0, "hkStringMap<hkResourceHandle*, hkContainerHeapAllocator>") // class hkStringMap< class hkResourceHandle*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkContainerResourceMap, s_libraryName, hkResourceMap)

#include <Common/Serialize/Serialize/Platform/hkPlatformObjectWriter.h>


// hkPlatformObjectWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPlatformObjectWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cache)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPlatformObjectWriter)
    HK_TRACKER_MEMBER(hkPlatformObjectWriter, m_copier, 0, "hkObjectCopier*") // class hkObjectCopier*
    HK_TRACKER_MEMBER(hkPlatformObjectWriter, m_cache, 0, "hkPlatformObjectWriter::Cache*") // class hkPlatformObjectWriter::Cache*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkPlatformObjectWriter, s_libraryName, hkObjectWriter)


// Cache hkPlatformObjectWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPlatformObjectWriter::Cache)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPlatformObjectWriter::Cache)
    HK_TRACKER_MEMBER(hkPlatformObjectWriter::Cache, m_platformClassFromHostClass, 0, "hkPointerMap<void*, void*, hkContainerHeapAllocator>") // class hkPointerMap< const void*, void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPlatformObjectWriter::Cache, m_platformClassComputed, 0, "hkPointerMap<hkClass*, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< const hkClass*, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPlatformObjectWriter::Cache, m_allocations, 0, "hkArray<void*, hkContainerHeapAllocator>") // hkArray< void*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkPlatformObjectWriter::Cache, s_libraryName, hkReferencedObject)

#include <Common/Serialize/Serialize/Xml/hkXmlObjectReader.h>


// hkXmlObjectReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlObjectReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlObjectReader)
    HK_TRACKER_MEMBER(hkXmlObjectReader, m_parser, 0, "hkXmlParser*") // class hkXmlParser*
    HK_TRACKER_MEMBER(hkXmlObjectReader, m_nameToObject, 0, "hkStringMap<void*, hkContainerHeapAllocator>*") // class hkStringMap< void*, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlObjectReader, s_libraryName, hkObjectReader)

#include <Common/Serialize/Serialize/Xml/hkXmlObjectWriter.h>


// hkXmlObjectWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlObjectWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NameFromAddress)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SequentialNameFromAddress)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlObjectWriter)
    HK_TRACKER_MEMBER(hkXmlObjectWriter, m_indent, 0, "hkInplaceArray<char, 16, hkContainerHeapAllocator>") // class hkInplaceArray< char, 16, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlObjectWriter, s_libraryName, hkObjectWriter)


// NameFromAddress hkXmlObjectWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlObjectWriter::NameFromAddress)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlObjectWriter::NameFromAddress)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkXmlObjectWriter::NameFromAddress, s_libraryName)


// SequentialNameFromAddress hkXmlObjectWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlObjectWriter::SequentialNameFromAddress)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlObjectWriter::SequentialNameFromAddress)
    HK_TRACKER_MEMBER(hkXmlObjectWriter::SequentialNameFromAddress, m_map, 0, "hkPointerMap<void*, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< const void*, hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlObjectWriter::SequentialNameFromAddress, s_libraryName, hkXmlObjectWriter::NameFromAddress)

#include <Common/Serialize/Serialize/hkObjectReader.h>


// hkObjectReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectReader)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkObjectReader, s_libraryName, hkReferencedObject)

#include <Common/Serialize/Serialize/hkObjectWriter.h>


// hkObjectWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkObjectWriter, s_libraryName, hkReferencedObject)

#include <Common/Serialize/Serialize/hkRelocationInfo.h>


// hkRelocationInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRelocationInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Local)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Global)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Finish)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Import)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRelocationInfo)
    HK_TRACKER_MEMBER(hkRelocationInfo, m_local, 0, "hkArray<hkRelocationInfo::Local, hkContainerHeapAllocator>") // hkArray< struct hkRelocationInfo::Local, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkRelocationInfo, m_global, 0, "hkArray<hkRelocationInfo::Global, hkContainerHeapAllocator>") // hkArray< struct hkRelocationInfo::Global, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkRelocationInfo, m_finish, 0, "hkArray<hkRelocationInfo::Finish, hkContainerHeapAllocator>") // hkArray< struct hkRelocationInfo::Finish, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkRelocationInfo, m_imports, 0, "hkArray<hkRelocationInfo::Import, hkContainerHeapAllocator>") // hkArray< struct hkRelocationInfo::Import, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkRelocationInfo, m_pool, 0, "hkStorageStringMap<hkInt32, hkContainerHeapAllocator>*") // class hkStorageStringMap< hkInt32, struct hkContainerHeapAllocator >*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRelocationInfo, s_libraryName)


// Local hkRelocationInfo
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkRelocationInfo, Local, s_libraryName)


// Global hkRelocationInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRelocationInfo::Global)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRelocationInfo::Global)
    HK_TRACKER_MEMBER(hkRelocationInfo::Global, m_toAddress, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkRelocationInfo::Global, m_toClass, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRelocationInfo::Global, s_libraryName)


// Finish hkRelocationInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRelocationInfo::Finish)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRelocationInfo::Finish)
    HK_TRACKER_MEMBER(hkRelocationInfo::Finish, m_className, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRelocationInfo::Finish, s_libraryName)


// Import hkRelocationInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRelocationInfo::Import)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRelocationInfo::Import)
    HK_TRACKER_MEMBER(hkRelocationInfo::Import, m_identifier, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRelocationInfo::Import, s_libraryName)

#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileCommon.h>


// Header hkBinaryTagfile

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryTagfile::Header)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryTagfile::Header)
    HK_TRACKER_MEMBER(hkBinaryTagfile::Header, m_sdk, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBinaryTagfile::Header, s_libraryName)

// hkBinaryTagfile TagType
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkBinaryTagfile::TagType, s_libraryName, hkBinaryTagfile_TagType)
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileReader.h>


// hkBinaryTagfileReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryTagfileReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryTagfileReader)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBinaryTagfileReader, s_libraryName, hkTagfileReader)

#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileWriter.h>


// hkBinaryTagfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryTagfileWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryTagfileWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBinaryTagfileWriter, s_libraryName, hkTagfileWriter)

#include <Common/Serialize/Tagfile/Text/hkTextTagfileWriter.h>


// hkTextTagfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTextTagfileWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTextTagfileWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTextTagfileWriter, s_libraryName, hkTagfileWriter)

#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileCommon.h>


// Header hkXmlTagfile

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlTagfile::Header)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlTagfile::Header)
    HK_TRACKER_MEMBER(hkXmlTagfile::Header, m_sdkVersion, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkXmlTagfile::Header, s_libraryName)

#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileReader.h>


// hkXmlTagfileReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlTagfileReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlTagfileReader)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlTagfileReader, s_libraryName, hkTagfileReader)

#include <Common/Serialize/Tagfile/Xml/hkXmlTagfileWriter.h>


// hkXmlTagfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlTagfileWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlTagfileWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlTagfileWriter, s_libraryName, hkTagfileWriter)

#include <Common/Serialize/Tagfile/hkTagfileReader.h>


// hkTagfileReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTagfileReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FormatType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTagfileReader)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkTagfileReader, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTagfileReader, FormatType, s_libraryName)

#include <Common/Serialize/Tagfile/hkTagfileWriter.h>


// hkTagfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTagfileWriter)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AddDataObjectListener)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Options)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTagfileWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkTagfileWriter, s_libraryName, hkReferencedObject)


// AddDataObjectListener hkTagfileWriter

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTagfileWriter::AddDataObjectListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTagfileWriter::AddDataObjectListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkTagfileWriter::AddDataObjectListener, s_libraryName, hkReferencedObject)


// Options hkTagfileWriter
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTagfileWriter, Options, s_libraryName)

#include <Common/Serialize/TypeManager/hkTypeManager.h>


// hkLegacyType ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkLegacyType, s_libraryName)


// hkTypeManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypeManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypeManager)
    HK_TRACKER_MEMBER(hkTypeManager, m_homogenousClass, 0, "hkTypeManager::Type*") // struct hkTypeManager::Type*
    HK_TRACKER_MEMBER(hkTypeManager, m_builtInTypes, 0, "hkTypeManager::Type* [10]") // struct hkTypeManager::Type* [10]
    HK_TRACKER_MEMBER(hkTypeManager, m_classMap, 0, "hkStorageStringMap<hkTypeManager::Type*, hkContainerHeapAllocator>") // class hkStorageStringMap< struct hkTypeManager::Type*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkTypeManager, m_typeMultiMap, 0, "hkPointerMultiMap<hkUint32, hkTypeManager::Type*>") // class hkPointerMultiMap< hkUint32, struct hkTypeManager::Type* >
    HK_TRACKER_MEMBER(hkTypeManager, m_typeFreeList, 0, "hkFreeList") // class hkFreeList
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkTypeManager, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkTypeManager, SubType, s_libraryName)


// Type hkTypeManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkTypeManager::Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkTypeManager::Type)
    HK_TRACKER_MEMBER(hkTypeManager::Type, m_parent, 0, "hkTypeManager::Type*") // struct hkTypeManager::Type*
    HK_TRACKER_MEMBER(hkTypeManager::Type, m_extra, 0, "hkTypeManager::Type::<anonymous>") // struct hkTypeManager::Type::<anonymous>
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkTypeManager::Type, s_libraryName)

#include <Common/Serialize/Util/Xml/hkFloatParseUtil.h>


// hkFloatParseUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkFloatParseUtil, s_libraryName)

#include <Common/Serialize/Util/Xml/hkParserBuffer.h>


// hkParserBuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkParserBuffer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkParserBuffer)
    HK_TRACKER_MEMBER(hkParserBuffer, m_pos, 0, "char*") // char*
    HK_TRACKER_MEMBER(hkParserBuffer, m_buffer, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkParserBuffer, m_reader, 0, "hkStreamReader*") // class hkStreamReader*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkParserBuffer, s_libraryName, hkReferencedObject)

#include <Common/Serialize/Util/Xml/hkXmlLexAnalyzer.h>


// hkXmlLexAnalyzer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlLexAnalyzer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Token)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlLexAnalyzer)
    HK_TRACKER_MEMBER(hkXmlLexAnalyzer, m_buffer, 0, "hkParserBuffer") // class hkParserBuffer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlLexAnalyzer, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkXmlLexAnalyzer, Token, s_libraryName)

#include <Common/Serialize/Util/Xml/hkXmlParser.h>


// hkXmlParser ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlParser)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Attribute)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Node)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(StartElement)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EndElement)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Characters)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NodeType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlParser)
    HK_TRACKER_MEMBER(hkXmlParser, m_pendingNodes, 0, "hkArray<hkXmlParser::Node*, hkContainerHeapAllocator>") // hkArray< struct hkXmlParser::Node*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlParser, m_lastError, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlParser, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkXmlParser, NodeType, s_libraryName)


// Attribute hkXmlParser

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlParser::Attribute)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlParser::Attribute)
    HK_TRACKER_MEMBER(hkXmlParser::Attribute, name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkXmlParser::Attribute, value, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkXmlParser::Attribute, s_libraryName)


// Node hkXmlParser

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlParser::Node)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlParser::Node)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlParser::Node, s_libraryName, hkReferencedObject)


// StartElement hkXmlParser

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlParser::StartElement)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlParser::StartElement)
    HK_TRACKER_MEMBER(hkXmlParser::StartElement, name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkXmlParser::StartElement, attributes, 0, "hkArray<hkXmlParser::Attribute, hkContainerHeapAllocator>") // hkArray< struct hkXmlParser::Attribute, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlParser::StartElement, s_libraryName, hkXmlParser::Node)


// EndElement hkXmlParser

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlParser::EndElement)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlParser::EndElement)
    HK_TRACKER_MEMBER(hkXmlParser::EndElement, name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlParser::EndElement, s_libraryName, hkXmlParser::Node)


// Characters hkXmlParser

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlParser::Characters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlParser::Characters)
    HK_TRACKER_MEMBER(hkXmlParser::Characters, text, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlParser::Characters, s_libraryName, hkXmlParser::Node)

#include <Common/Serialize/Util/Xml/hkXmlStreamParser.h>


// hkXmlStreamParser ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlStreamParser)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SubString)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Token)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlStreamParser)
    HK_TRACKER_MEMBER(hkXmlStreamParser, m_attribMap, 0, "hkStringMap<hkInt32, hkContainerHeapAllocator>") // class hkStringMap< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlStreamParser, m_keys, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlStreamParser, m_keyStorage, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlStreamParser, m_lex, 0, "hkXmlLexAnalyzer") // class hkXmlLexAnalyzer
    HK_TRACKER_MEMBER(hkXmlStreamParser, m_lexemes, 0, "hkArray<hkXmlStreamParser::SubString, hkContainerHeapAllocator>") // hkArray< struct hkXmlStreamParser::SubString, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlStreamParser, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkXmlStreamParser, Token, s_libraryName)


// SubString hkXmlStreamParser
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkXmlStreamParser, SubString, s_libraryName)

#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>


// hkBuiltinTypeRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBuiltinTypeRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBuiltinTypeRegistry)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkBuiltinTypeRegistry, s_libraryName, hkReferencedObject)

#include <Common/Serialize/Util/hkChainedClassNameRegistry.h>


// hkChainedClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkChainedClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkChainedClassNameRegistry)
    HK_TRACKER_MEMBER(hkChainedClassNameRegistry, m_nextRegistry, 0, "hkClassNameRegistry*") // const class hkClassNameRegistry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkChainedClassNameRegistry, s_libraryName, hkDynamicClassNameRegistry)

#include <Common/Serialize/Util/hkClassPointerVtable.h>


// TypeInfoRegistry hkClassPointerVtable

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassPointerVtable::TypeInfoRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassPointerVtable::TypeInfoRegistry)
    HK_TRACKER_MEMBER(hkClassPointerVtable::TypeInfoRegistry, m_classes, 0, "hkClassNameRegistry *") // const class hkClassNameRegistry *
    HK_TRACKER_MEMBER(hkClassPointerVtable::TypeInfoRegistry, m_typeInfos, 0, "hkStringMap<hkTypeInfo*, hkContainerHeapAllocator>") // class hkStringMap< class hkTypeInfo*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkClassPointerVtable::TypeInfoRegistry, s_libraryName, hkTypeInfoRegistry)


// VtableRegistry hkClassPointerVtable

HK_TRACKER_DECLARE_CLASS_BEGIN(hkClassPointerVtable::VtableRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkClassPointerVtable::VtableRegistry)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkClassPointerVtable::VtableRegistry, s_libraryName, hkVtableClassRegistry)

#include <Common/Serialize/Util/hkLoader.h>


// hkLoader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkLoader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkLoader)
    HK_TRACKER_MEMBER(hkLoader, m_loadedData, 0, "hkArray<hkResource*, hkContainerHeapAllocator>") // hkArray< class hkResource*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkLoader, s_libraryName, hkReferencedObject)

#include <Common/Serialize/Util/hkObjectInspector.h>


// Pointer hkObjectInspector

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectInspector::Pointer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectInspector::Pointer)
    HK_TRACKER_MEMBER(hkObjectInspector::Pointer, location, 0, "void**") // void**
    HK_TRACKER_MEMBER(hkObjectInspector::Pointer, klass, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkObjectInspector::Pointer, s_libraryName)


// ObjectListener hkObjectInspector

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectInspector::ObjectListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectInspector::ObjectListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkObjectInspector::ObjectListener, s_libraryName)

#include <Common/Serialize/Util/hkRootLevelContainer.h>


// hkRootLevelContainer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRootLevelContainer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(NamedVariant)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRootLevelContainer)
    HK_TRACKER_MEMBER(hkRootLevelContainer, m_namedVariants, 0, "hkArray<hkRootLevelContainer::NamedVariant, hkContainerHeapAllocator>") // hkArray< class hkRootLevelContainer::NamedVariant, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRootLevelContainer, s_libraryName)


// NamedVariant hkRootLevelContainer

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRootLevelContainer::NamedVariant)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRootLevelContainer::NamedVariant)
    HK_TRACKER_MEMBER(hkRootLevelContainer::NamedVariant, m_name, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkRootLevelContainer::NamedVariant, m_className, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkRootLevelContainer::NamedVariant, m_variant, 0, "hkReferencedObject *") // class hkReferencedObject *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkRootLevelContainer::NamedVariant, s_libraryName)

#include <Common/Serialize/Util/hkSerializationCheckingUtils.h>


// DeferredErrorStream hkSerializationCheckingUtils

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSerializationCheckingUtils::DeferredErrorStream)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSerializationCheckingUtils::DeferredErrorStream)
    HK_TRACKER_MEMBER(hkSerializationCheckingUtils::DeferredErrorStream, m_data, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSerializationCheckingUtils::DeferredErrorStream, s_libraryName, hkStreamWriter)

#include <Common/Serialize/Util/hkSerializeDeprecated.h>


// hkSerializeDeprecated ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSerializeDeprecated)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(XmlPackfileHeader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSerializeDeprecated)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSerializeDeprecated, s_libraryName, hkReferencedObject)


// XmlPackfileHeader hkSerializeDeprecated

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSerializeDeprecated::XmlPackfileHeader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSerializeDeprecated::XmlPackfileHeader)
    HK_TRACKER_MEMBER(hkSerializeDeprecated::XmlPackfileHeader, m_contentsVersion, 0, "hkStringPtr") // hkStringPtr
    HK_TRACKER_MEMBER(hkSerializeDeprecated::XmlPackfileHeader, m_topLevelObject, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSerializeDeprecated::XmlPackfileHeader, s_libraryName)

#include <Common/Serialize/Util/hkSerializeUtil.h>


// ErrorDetails hkSerializeUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSerializeUtil::ErrorDetails)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ErrorID)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSerializeUtil::ErrorDetails)
    HK_TRACKER_MEMBER(hkSerializeUtil::ErrorDetails, defaultMessage, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSerializeUtil::ErrorDetails, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkSerializeUtil::ErrorDetails, ErrorID, s_libraryName)


// SaveOptions hkSerializeUtil
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkSerializeUtil::SaveOptions, s_libraryName, hkSerializeUtil_SaveOptions)


// LoadOptions hkSerializeUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSerializeUtil::LoadOptions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSerializeUtil::LoadOptions)
    HK_TRACKER_MEMBER(hkSerializeUtil::LoadOptions, m_classNameReg, 0, "hkClassNameRegistry*") // const class hkClassNameRegistry*
    HK_TRACKER_MEMBER(hkSerializeUtil::LoadOptions, m_typeInfoReg, 0, "hkTypeInfoRegistry*") // const class hkTypeInfoRegistry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkSerializeUtil::LoadOptions, s_libraryName, hkFlagsenumhkSerializeUtilLoadOptionBitsint)


// FormatDetails hkSerializeUtil

HK_TRACKER_DECLARE_CLASS_BEGIN(hkSerializeUtil::FormatDetails)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkSerializeUtil::FormatDetails)
    HK_TRACKER_MEMBER(hkSerializeUtil::FormatDetails, m_version, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkSerializeUtil::FormatDetails, s_libraryName)

// hkSerializeUtil SaveOptionBits
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkSerializeUtil::SaveOptionBits, s_libraryName, hkSerializeUtil_SaveOptionBits)
// hkSerializeUtil LoadOptionBits
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkSerializeUtil::LoadOptionBits, s_libraryName, hkSerializeUtil_LoadOptionBits)
// hkSerializeUtil FormatType
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkSerializeUtil::FormatType, s_libraryName, hkSerializeUtil_FormatType)
#include <Common/Serialize/Util/hkStaticClassNameRegistry.h>


// hkStaticClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStaticClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStaticClassNameRegistry)
    HK_TRACKER_MEMBER(hkStaticClassNameRegistry, m_name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkStaticClassNameRegistry, m_classes, 0, "hkClass**") // const const hkClass**
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkStaticClassNameRegistry, s_libraryName, hkClassNameRegistry)

#include <Common/Serialize/Util/hkStructureLayout.h>


// hkStructureLayout ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStructureLayout)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LayoutRules)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkStructureLayout, s_libraryName)


// LayoutRules hkStructureLayout
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkStructureLayout, LayoutRules, s_libraryName)

#include <Common/Serialize/Util/hkVersionCheckingUtils.h>

// hkVersionCheckingUtils Flags
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVersionCheckingUtils::Flags, s_libraryName, hkVersionCheckingUtils_Flags)
#include <Common/Serialize/Version/hkVersionPatchManager.h>


// hkVersionPatchManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PatchInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DependsPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemberRenamedPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemberAddedPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MemberRemovedPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SetParentPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FunctionPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CastPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DefaultChangedPatch)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClassWrapper)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UidFromClassVersion)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PatchType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager)
    HK_TRACKER_MEMBER(hkVersionPatchManager, m_uidFromClassVersion, 0, "hkVersionPatchManager::UidFromClassVersion*") // class hkVersionPatchManager::UidFromClassVersion*
    HK_TRACKER_MEMBER(hkVersionPatchManager, m_patchInfos, 0, "hkArray<hkVersionPatchManager::PatchInfo*, hkContainerHeapAllocator>") // hkArray< const struct hkVersionPatchManager::PatchInfo*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVersionPatchManager, m_patchIndexFromUid, 0, "hkPointerMap<hkUint64, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkVersionPatchManager, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVersionPatchManager, PatchType, s_libraryName)


// PatchInfo hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::PatchInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Component)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::PatchInfo)
    HK_TRACKER_MEMBER(hkVersionPatchManager::PatchInfo, oldName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::PatchInfo, newName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::PatchInfo, s_libraryName)


// Component hkVersionPatchManager::PatchInfo

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::PatchInfo::Component)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::PatchInfo::Component)
    HK_TRACKER_MEMBER(hkVersionPatchManager::PatchInfo::Component, patch, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::PatchInfo::Component, s_libraryName)


// DependsPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::DependsPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::DependsPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::DependsPatch, name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::DependsPatch, s_libraryName)


// MemberRenamedPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::MemberRenamedPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::MemberRenamedPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberRenamedPatch, oldName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberRenamedPatch, newName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::MemberRenamedPatch, s_libraryName)


// MemberAddedPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::MemberAddedPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::MemberAddedPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberAddedPatch, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberAddedPatch, typeName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberAddedPatch, defaultPtr, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::MemberAddedPatch, s_libraryName)


// MemberRemovedPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::MemberRemovedPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::MemberRemovedPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberRemovedPatch, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::MemberRemovedPatch, typeName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::MemberRemovedPatch, s_libraryName)


// SetParentPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::SetParentPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::SetParentPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::SetParentPatch, oldParent, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::SetParentPatch, newParent, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::SetParentPatch, s_libraryName)


// FunctionPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::FunctionPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::FunctionPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::FunctionPatch, name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::FunctionPatch, s_libraryName)


// CastPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::CastPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::CastPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::CastPatch, name, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::CastPatch, s_libraryName)


// DefaultChangedPatch hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::DefaultChangedPatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::DefaultChangedPatch)
    HK_TRACKER_MEMBER(hkVersionPatchManager::DefaultChangedPatch, name, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionPatchManager::DefaultChangedPatch, defaultPtr, 0, "void*") // const void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::DefaultChangedPatch, s_libraryName)


// ClassWrapper hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::ClassWrapper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::ClassWrapper)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkVersionPatchManager::ClassWrapper, s_libraryName, hkReferencedObject)


// UidFromClassVersion hkVersionPatchManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionPatchManager::UidFromClassVersion)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionPatchManager::UidFromClassVersion)
    HK_TRACKER_MEMBER(hkVersionPatchManager::UidFromClassVersion, m_indexFromName, 0, "hkStringMap<hkInt32, hkContainerHeapAllocator>") // class hkStringMap< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVersionPatchManager::UidFromClassVersion, m_names, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVersionPatchManager::UidFromClassVersion, m_cachedNames, 0, "hkStringMap<char*, hkContainerHeapAllocator>") // class hkStringMap< char*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionPatchManager::UidFromClassVersion, s_libraryName)


// hkDefaultClassWrapper ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDefaultClassWrapper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDefaultClassWrapper)
    HK_TRACKER_MEMBER(hkDefaultClassWrapper, m_nameReg, 0, "hkClassNameRegistry *") // const class hkClassNameRegistry *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDefaultClassWrapper, s_libraryName, hkVersionPatchManager::ClassWrapper)

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
