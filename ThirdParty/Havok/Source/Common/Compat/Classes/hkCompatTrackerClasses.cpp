/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Compat/hkCompat.h>
static const char s_libraryName[] = "hkCompat";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkCompatRegister() {}

#include <Common/Compat/Deprecated/Packfile/Binary/hkBinaryPackfileReader.h>


// hkBinaryPackfileReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryPackfileReader)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(PackfileObject)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(BinaryPackfileData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SectionHeaderArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryPackfileReader)
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_packfileData, 0, "hkBinaryPackfileReader::BinaryPackfileData*") // class hkBinaryPackfileReader::BinaryPackfileData*
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_header, 0, "hkPackfileHeader*") // class hkPackfileHeader*
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_sections, 0, "hkBinaryPackfileReader::SectionHeaderArray") // class hkBinaryPackfileReader::SectionHeaderArray
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_sectionData, 0, "hkInplaceArray<void*, 16, hkContainerHeapAllocator>") // class hkInplaceArray< void*, 16, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_loadedObjects, 0, "hkArray<hkVariant, hkContainerHeapAllocator>*") // hkArray< hkVariant, struct hkContainerHeapAllocator >*
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_tracker, 0, "hkPackfileObjectUpdateTracker*") // class hkPackfileObjectUpdateTracker*
    HK_TRACKER_MEMBER(hkBinaryPackfileReader, m_packfileClassRegistry, 0, "hkChainedClassNameRegistry *") // class hkChainedClassNameRegistry *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBinaryPackfileReader, s_libraryName, hkPackfileReader)


// PackfileObject hkBinaryPackfileReader

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryPackfileReader::PackfileObject)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryPackfileReader::PackfileObject)
    HK_TRACKER_MEMBER(hkBinaryPackfileReader::PackfileObject, object, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBinaryPackfileReader::PackfileObject, s_libraryName)


// BinaryPackfileData hkBinaryPackfileReader

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryPackfileReader::BinaryPackfileData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryPackfileReader::BinaryPackfileData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBinaryPackfileReader::BinaryPackfileData, s_libraryName, hkPackfileData)


// SectionHeaderArray hkBinaryPackfileReader

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBinaryPackfileReader::SectionHeaderArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBinaryPackfileReader::SectionHeaderArray)
    HK_TRACKER_MEMBER(hkBinaryPackfileReader::SectionHeaderArray, m_baseSection, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBinaryPackfileReader::SectionHeaderArray, s_libraryName)

#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileReader.h>


// hkXmlPackfileReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlPackfileReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlPackfileReader)
    HK_TRACKER_MEMBER(hkXmlPackfileReader, m_data, 0, "hkPackfileData*") // class hkPackfileData*
    HK_TRACKER_MEMBER(hkXmlPackfileReader, m_knownSections, 0, "hkArray<char*, hkContainerHeapAllocator>") // hkArray< const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlPackfileReader, m_sectionTagToIndex, 0, "hkStringMap<hkInt32, hkContainerHeapAllocator>") // class hkStringMap< hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlPackfileReader, m_loadedObjects, 0, "hkArray<hkVariant, hkContainerHeapAllocator>") // hkArray< hkVariant, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkXmlPackfileReader, m_tracker, 0, "hkXmlPackfileUpdateTracker*") // class hkXmlPackfileUpdateTracker*
    HK_TRACKER_MEMBER(hkXmlPackfileReader, m_packfileClassRegistry, 0, "hkChainedClassNameRegistry *") // class hkChainedClassNameRegistry *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlPackfileReader, s_libraryName, hkPackfileReader)

#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileWriter.h>


// hkXmlPackfileWriter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkXmlPackfileWriter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkXmlPackfileWriter)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkXmlPackfileWriter, s_libraryName, hkPackfileWriter)

#include <Common/Compat/Deprecated/Packfile/hkPackfileReader.h>


// hkPackfileReader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileReader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileReader)
    HK_TRACKER_MEMBER(hkPackfileReader, m_updateFlagFromClass, 0, "hkPointerMap<hkClass*, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< const hkClass*, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileReader, m_contentsVersion, 0, "char*") // char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkPackfileReader, s_libraryName, hkReferencedObject)

#include <Common/Compat/Deprecated/Util/hkBindingClassNameRegistry.h>


// hkBindingClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBindingClassNameRegistry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClassAllocationsTracker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBindingClassNameRegistry)
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry, m_tracker, 0, "hkBindingClassNameRegistry::ClassAllocationsTracker") // class hkBindingClassNameRegistry::ClassAllocationsTracker
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry, m_newNameFromOldName, 0, "hkStringMap<char*, hkContainerHeapAllocator>") // class hkStringMap< const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry, m_oldNameFromNewName, 0, "hkStringMap<char*, hkContainerHeapAllocator>") // class hkStringMap< const char*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkBindingClassNameRegistry, s_libraryName, hkChainedClassNameRegistry)


// ClassAllocationsTracker hkBindingClassNameRegistry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBindingClassNameRegistry::ClassAllocationsTracker)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClassData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UnresolvedClassPointerTracker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBindingClassNameRegistry::ClassAllocationsTracker)
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry::ClassAllocationsTracker, m_trackedClassData, 0, "hkArray<hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData, hkContainerHeapAllocator>") // hkArray< struct hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBindingClassNameRegistry::ClassAllocationsTracker, s_libraryName)


// ClassData hkBindingClassNameRegistry::ClassAllocationsTracker

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData)
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData, allocatedClass, 0, "hkClass*") // hkClass*
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData, enumsAllocatedForMembers, 0, "hkPointerMap<hkInternalClassEnum*, hkUint32, hkContainerHeapAllocator>") // class hkPointerMap< struct hkInternalClassEnum*, hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBindingClassNameRegistry::ClassAllocationsTracker::ClassData, s_libraryName)


// UnresolvedClassPointerTracker hkBindingClassNameRegistry::ClassAllocationsTracker

HK_TRACKER_DECLARE_CLASS_BEGIN(hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker)
    HK_TRACKER_MEMBER(hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker, m_pointers, 0, "hkSerializeMultiMap<void*, hkClass**, hkPointerMap<void*, hkInt32, hkContainerHeapAllocator> >") // class hkSerializeMultiMap< void*, const hkClass**, class hkPointerMap< void*, hkInt32, struct hkContainerHeapAllocator > >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkBindingClassNameRegistry::ClassAllocationsTracker::UnresolvedClassPointerTracker, s_libraryName)

#include <Common/Compat/Deprecated/Util/hkRenamedClassNameRegistry.h>


// hkRenamedClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRenamedClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRenamedClassNameRegistry)
    HK_TRACKER_MEMBER(hkRenamedClassNameRegistry, m_renames, 0, "hkStringMap<char*, hkContainerHeapAllocator>") // class hkStringMap< const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkRenamedClassNameRegistry, m_originalRegistry, 0, "hkClassNameRegistry*") // const class hkClassNameRegistry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkRenamedClassNameRegistry, s_libraryName, hkClassNameRegistry)

#include <Common/Compat/Deprecated/Version/hkObjectUpdateTracker.h>


// hkObjectUpdateTracker ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectUpdateTracker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectUpdateTracker)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkObjectUpdateTracker, s_libraryName, hkReferencedObject)

#include <Common/Compat/Deprecated/Version/hkPackfileObjectUpdateTracker.h>


// hkPackfileObjectUpdateTracker ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkPackfileObjectUpdateTracker)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkPackfileObjectUpdateTracker)
    HK_TRACKER_MEMBER(hkPackfileObjectUpdateTracker, m_packfileData, 0, "hkPackfileData*") // class hkPackfileData*
    HK_TRACKER_MEMBER(hkPackfileObjectUpdateTracker, m_pointers, 0, "hkSerializeMultiMap<void*, void*, hkPointerMap<void*, hkInt32, hkContainerHeapAllocator> >") // class hkSerializeMultiMap< void*, void*, class hkPointerMap< void*, hkInt32, struct hkContainerHeapAllocator > >
    HK_TRACKER_MEMBER(hkPackfileObjectUpdateTracker, m_finish, 0, "hkPointerMap<void*, char*, hkContainerHeapAllocator>") // class hkPointerMap< void*, const char*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkPackfileObjectUpdateTracker, m_topLevelObject, 0, "void*") // void*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkPackfileObjectUpdateTracker, s_libraryName, hkObjectUpdateTracker)

#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>


// hkVersionRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionRegistry)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClassRename)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ClassAction)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UpdateDescription)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Updater)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SignatureFlags)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VersionFlags)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionRegistry)
    HK_TRACKER_MEMBER(hkVersionRegistry, m_updaters, 0, "hkArray<hkVersionRegistry::Updater*, hkContainerHeapAllocator>") // hkArray< const struct hkVersionRegistry::Updater*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVersionRegistry, m_versionToClassNameRegistryMap, 0, "hkStringMap<hkDynamicClassNameRegistry*, hkContainerHeapAllocator>") // class hkStringMap< class hkDynamicClassNameRegistry*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkVersionRegistry, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVersionRegistry, SignatureFlags, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVersionRegistry, VersionFlags, s_libraryName)


// ClassRename hkVersionRegistry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionRegistry::ClassRename)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionRegistry::ClassRename)
    HK_TRACKER_MEMBER(hkVersionRegistry::ClassRename, oldName, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionRegistry::ClassRename, newName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionRegistry::ClassRename, s_libraryName)


// ClassAction hkVersionRegistry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionRegistry::ClassAction)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionRegistry::ClassAction)
    HK_TRACKER_MEMBER(hkVersionRegistry::ClassAction, oldClassName, 0, "char*") // const char*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionRegistry::ClassAction, s_libraryName)


// UpdateDescription hkVersionRegistry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionRegistry::UpdateDescription)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionRegistry::UpdateDescription)
    HK_TRACKER_MEMBER(hkVersionRegistry::UpdateDescription, m_renames, 0, "hkVersionRegistry::ClassRename*") // const struct hkVersionRegistry::ClassRename*
    HK_TRACKER_MEMBER(hkVersionRegistry::UpdateDescription, m_actions, 0, "hkVersionRegistry::ClassAction*") // const struct hkVersionRegistry::ClassAction*
    HK_TRACKER_MEMBER(hkVersionRegistry::UpdateDescription, m_newClassRegistry, 0, "hkClassNameRegistry*") // const class hkClassNameRegistry*
    HK_TRACKER_MEMBER(hkVersionRegistry::UpdateDescription, m_next, 0, "hkVersionRegistry::UpdateDescription*") // struct hkVersionRegistry::UpdateDescription*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionRegistry::UpdateDescription, s_libraryName)


// Updater hkVersionRegistry

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVersionRegistry::Updater)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVersionRegistry::Updater)
    HK_TRACKER_MEMBER(hkVersionRegistry::Updater, fromVersion, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionRegistry::Updater, toVersion, 0, "char*") // const char*
    HK_TRACKER_MEMBER(hkVersionRegistry::Updater, desc, 0, "hkVersionRegistry::UpdateDescription*") // struct hkVersionRegistry::UpdateDescription*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVersionRegistry::Updater, s_libraryName)


// ValidatedClassNameRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(ValidatedClassNameRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(ValidatedClassNameRegistry)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(ValidatedClassNameRegistry, s_libraryName, hkDynamicClassNameRegistry)

#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>


// CollectClassDefinitions ::

HK_TRACKER_DECLARE_CLASS_BEGIN(CollectClassDefinitions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(CollectClassDefinitions)
    HK_TRACKER_MEMBER(CollectClassDefinitions, m_classMemberEnumType, 0, "hkClassEnum*") // const class hkClassEnum*
    HK_TRACKER_MEMBER(CollectClassDefinitions, m_classExternList, 0, "hkStringBuf") // class hkStringBuf
    HK_TRACKER_MEMBER(CollectClassDefinitions, m_classDefinitionList, 0, "hkStringBuf") // class hkStringBuf
    HK_TRACKER_MEMBER(CollectClassDefinitions, m_doneClasses, 0, "hkStringMap<hkUint32, hkContainerHeapAllocator>") // class hkStringMap< hkUint32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(CollectClassDefinitions, s_libraryName)

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
