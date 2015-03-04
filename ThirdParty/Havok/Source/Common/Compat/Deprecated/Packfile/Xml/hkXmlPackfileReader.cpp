/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Compat/Deprecated/Packfile/Xml/hkXmlPackfileReader.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/StringMap/hkStorageStringMap.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Serialize/Serialize/Xml/hkXmlObjectReader.h>
#include <Common/Serialize/Util/hkChainedClassNameRegistry.h>
#include <Common/Serialize/Util/hkStructureLayout.h>
#include <Common/Serialize/Util/Xml/hkXmlParser.h>
#include <Common/Compat/Deprecated/Version/hkPackfileObjectUpdateTracker.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>

extern const hkClass hkClassVersion1PaddedClass;
extern const hkClass hkClassVersion2PaddedClass;
extern const hkClass hkClassEnumVersion1PaddedClass;

extern const hkClass hkClassVersion3Class;
extern const hkClass hkClassMemberVersion3Class;

#if 0
static int tab = 0;
#	define REPORT(INDENT, TEXT) { \
	char reportBuf[512]; \
	hkErrStream ostr(reportBuf,sizeof(reportBuf)); \
	int indent = INDENT; while(indent-- > 0){ostr << "\t";}; \
	ostr << TEXT; \
	hkError::getInstance().message(hkError::MESSAGE_REPORT, 0, reportBuf, "XML", 0); \
}

#	define TRACE_FUNC_ENTER(A) REPORT(tab++, A)
#	define TRACE_FUNC_EXIT(A) REPORT(--tab, A)
#	define TRACE(A) REPORT(tab, A)
#else
#	define TRACE_FUNC_ENTER(A)
#	define TRACE_FUNC_EXIT(A)
#	define TRACE(A)
#endif

namespace
{
#if defined(HK_DEBUG)
	// hkClass signature fix. HVK-3680
	static inline bool xmlPackFileReader_containsCorrectSignatures(const char* contentsVersion)
	{
		// These are wildcards because of the way string matching is done
		static const char* versionsWithSignatureBug[] =
		{
			"Havok-3",
			"Havok-4.0",
			"Havok-4.1",
			"Havok-4.5.0-b1",
			"Havok-4.6.0-r1",
			HK_NULL
		};
		for( int i = 0; versionsWithSignatureBug[i] != HK_NULL; ++i )
		{
			const char* ver = versionsWithSignatureBug[i];
			if( hkString::strNcmp(contentsVersion, ver, hkString::strLen(ver) ) == 0)
			{
				return false;
			}
		}
		return true;
	}
#endif

	static inline bool mayContainWrongMetadata(const char* contentsVersion)
	{
		// These are wildcards because of the way string matching is done.
		static const char* versionsWithWrongMetadataBug[] =
		{
			"Havok-5.0.0-b1",
			"Havok-5.0.0-r1",
			"Havok-5.1.0-r1",
			HK_NULL
		};
		for( int i = 0; versionsWithWrongMetadataBug[i] != HK_NULL; ++i )
		{
			const char* ver = versionsWithWrongMetadataBug[i];
			if( hkString::strNcmp(contentsVersion, ver, hkString::strLen(ver) ) == 0)
			{
				return true;
			}
		}
		return false;
	}

	static inline bool xmlPackFileReader_IsPublicClassObject(const char* klassName)
	{
		return ( klassName
			&& hkString::strCmp(klassName, "hkClass") != 0
			&& hkString::strCmp(klassName, "hkClassMember") != 0
			&& hkString::strCmp(klassName, "hkClassEnum") != 0
			&& hkString::strCmp(klassName, "hkClassEnumItem") != 0 );
	}

	class hkPatchClassInstanceXmlParser : public hkXmlParser
	{
		public:
			hkPatchClassInstanceXmlParser() :
				m_classInstanceNodesCount(0),
				m_allowTopLevelClassInstancesOnly(false),
				m_behaveAsParentCount(1) { }

			virtual hkResult nextNode( Node** nodeOut, hkStreamReader* reader )
			{
				hkResult res = hkXmlParser::nextNode( nodeOut, reader );
				if( res == HK_SUCCESS && m_behaveAsParentCount == 0 )
				{
					hkXmlParser::StartElement* start = (*nodeOut)->asStart();
					if( start )
					{
						const char* className = start->getAttribute("class", HK_NULL );
						if( m_classInstanceNodesCount > 0
							|| ( start->name == "hkobject" && !xmlPackFileReader_IsPublicClassObject(className) ) )
						{
							++m_classInstanceNodesCount;
							if( start->name == "hkparam" )
							{
								const char* paramName = start->getAttribute("name", HK_NULL);
								HK_ASSERT2(0x6088e490, paramName != HK_NULL, "Found XML element with no 'name' attribute." );
								if( hkString::strCmp("attributes", paramName) == 0 )
								{
									--m_classInstanceNodesCount;
									res = getNextCorrectNode( start, nodeOut, reader );
								}
							}
						}
						else if( m_allowTopLevelClassInstancesOnly && start->name == "hkobject" && xmlPackFileReader_IsPublicClassObject(className) )
						{
							res = getNextCorrectNode( start, nodeOut, reader );
						}
					}
					else if( (*nodeOut)->asEnd() && m_classInstanceNodesCount > 0 )
					{
						--m_classInstanceNodesCount;
					}
				}
				return res;
			}

			virtual void putBack( Node* node )
			{
				if( m_classInstanceNodesCount > 0 )
				{
					if( node->asStart() )
					{
						--m_classInstanceNodesCount;
					}
					else if( node->asEnd() )
					{
						++m_classInstanceNodesCount;
					}
				}
				hkXmlParser::putBack( node );
			}

			virtual hkResult expandNode( StartElement* s, Tree& tree, hkStreamReader* reader )
			{
				++m_behaveAsParentCount; // increase to make it behaving as parent
				hkResult res = hkXmlParser::expandNode(s, tree, reader);
				--m_behaveAsParentCount; // decrease to return to behavior set previously
				return res;
			}

			void allowClassInstancesOnly( hkBool32 classInstancesOnly )
			{
				HK_ASSERT(0x256758e7, m_classInstanceNodesCount == 0); // this must be called only at 'hksection' tag level
				m_allowTopLevelClassInstancesOnly = classInstancesOnly;
				// check start/stop processing xml node by decreasing/increasing the count correspondingly
				m_behaveAsParentCount += m_allowTopLevelClassInstancesOnly ? -1 : 1;
			}

		private:

			inline hkResult getNextCorrectNode( StartElement* start, Node** nodeOut, hkStreamReader* reader )
			{
				Tree tree;
				hkResult res = expandNode(start, tree, reader);
				start->removeReference(); // now owned by tree
				if( res == HK_SUCCESS )
				{
					return nextNode( nodeOut, reader );
				}
				return res;
			}

			hkUint32 m_classInstanceNodesCount;
			hkBool32 m_allowTopLevelClassInstancesOnly;
			hkUint32 m_behaveAsParentCount; // 1 (default), '> 0' - behave as parent, '== 0' - process xml nodes 
	};

	typedef hkPointerMap<const void*, const hkClass*> ClassFromAddressMap;

	// COM-434, fix hkClassEnumClass pointers for
	// hkxSparselyAnimatedEnumClass and hctAttributeDescriptionClass
	static void fixClassEnumClassPointers(hkClass& klass)
	{
		if( hkString::strCmp(klass.getName(), "hkxSparselyAnimatedEnum") == 0
			|| hkString::strCmp(klass.getName(), "hctAttributeDescription") == 0 )
		{
			const char* memberNameToFix = hkString::strCmp(klass.getName(), "hkxSparselyAnimatedEnum") == 0 ? "type" : "enum";
			hkInternalClassMember* m = reinterpret_cast<hkInternalClassMember*>(const_cast<hkClassMember*>(klass.getMemberByName(memberNameToFix)));
			HK_ASSERT(0x45ae0d4d, m);
			if( m->m_class == HK_NULL )
			{
				// fix hkClassEnumClass pointer for
				// hkxSparselyAnimatedEnum::m_type and hctAttributeDescription::m_enum
				HK_ASSERT(0x45ae0d4e, m->m_type == hkClassMember::TYPE_POINTER && m->m_subtype == hkClassMember::TYPE_STRUCT);
				m->m_class = &hkClassEnumClass;
			}
		}
	}

	static inline const hkClassNameRegistry* getGlobalRegistryForPackfileVersion(const char* version)
	{
		if( version == HK_NULL )
		{
			return HK_NULL;
		}
		if( hkString::strCmp(hkVersionUtil::getCurrentVersion(), version) == 0 )
		{
			return hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
		}

		return hkVersionRegistry::getInstance().getClassNameRegistry(version);
	}

	// COM-608
	static inline void validateAndRegisterClasses(hkDynamicClassNameRegistry& registry, const hkClass* topClass)
	{
		if( topClass == HK_NULL || registry.getClassByName(topClass->getName()) )
		{
			return;
		}
		registry.registerClass(topClass);
		validateAndRegisterClasses(registry, topClass->getParent());
		for( int i = 0; i < topClass->getNumDeclaredMembers(); ++i )
		{
			const hkClassMember& m = topClass->getDeclaredMember(i);
			validateAndRegisterClasses(registry, m.getClass());
		}
	}

	// COM-608
	static void findAndRegisterHomogeneousClassesIn(hkDynamicClassNameRegistry& registry, const void* pointer, const hkClass& c, int numObjs)
	{
		for( int i = 0; i < c.getNumDeclaredMembers(); ++i )
		{
			const hkClassMember& m = c.getDeclaredMember(i);
			if( m.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
			{
				continue;
			}
			switch( m.getType() )
			{
				case hkClassMember::TYPE_STRUCT:
				{
					HK_ASSERT(0x13271ead, m.hasClass());
					int nelem = m.getCstyleArraySize() ? m.getCstyleArraySize() : 1;
					const void* o = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor structMember(const_cast<void*>(o), &m);
						findAndRegisterHomogeneousClassesIn(registry, structMember.asRaw(), m.getStructClass(), nelem);
						o = hkAddByteOffsetConst(o, c.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( m.getSubType() == hkClassMember::TYPE_STRUCT )
					{
						HK_ASSERT(0x735ff719, m.hasClass());
						const void* o = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor arrayMember(const_cast<void*>(o), &m);
							hkClassMemberAccessor::SimpleArray& sa = arrayMember.asSimpleArray();
							if( sa.data && sa.size )
							{
								findAndRegisterHomogeneousClassesIn(registry, sa.data, m.getStructClass(), sa.size);
							}
							o = hkAddByteOffsetConst(o, c.getObjectSize());
						}
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					const void* o = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor arrayMember(const_cast<void*>(o), &m);
						hkClassMemberAccessor::HomogeneousArray& ha = arrayMember.asHomogeneousArray();
						validateAndRegisterClasses(registry, ha.klass);
						if( ha.klass && ha.data && ha.size )
						{
							findAndRegisterHomogeneousClassesIn(registry, ha.data, *ha.klass, ha.size);
						}
						o = hkAddByteOffsetConst(o, c.getObjectSize());
					}
					break;
				}
				default:
				{
					// skip over all other types
				}
			}
		}
	}
}

// 
// COM-608
// Find and track all packfile hkClass pointers.
// Invalidate class registry when tracker is updated (replaceObject()/removeFinish()).
//
class hkXmlPackfileUpdateTracker : public hkPackfileObjectUpdateTracker
{
	inline void setVariantClassPointers( hkVariant* v, int numVariants, const ClassFromAddressMap& classFromObject )
	{
		for( int i = 0; i < numVariants; ++i )
		{
			if( v[i].m_object && v[i].m_class == HK_NULL )
			{
				v[i].m_class = classFromObject.getWithDefault(v[i].m_object, HK_NULL);
				if( v[i].m_class == HK_NULL )
				{
					HK_WARN_ALWAYS(0x067fde46, "Can not find class pointer for an object at 0x" << v[i].m_object << ".\n"
						<< "You will have to set manually corresponding class pointer in the variant. Otherwise you have to store metadata in the packfile.");
				}
				else
				{
					objectPointedBy(const_cast<hkClass*>(v[i].m_class), &v[i].m_class);
				}
			}
		}
	}

	// resurrect and track hkVariant classes, track homogeneous array classes
	void retrackVariantAndHomogeneousClasses( void* pointer, const hkClass& klass, const ClassFromAddressMap& classFromObject, int numObjs )
	{
		for( int memberIndex = 0; memberIndex < klass.getNumMembers(); ++memberIndex )
		{
			const hkClassMember& member = klass.getMember(memberIndex);
			if( member.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
			{
				continue;
			}
			switch( member.getType() )
			{
				case hkClassMember::TYPE_VARIANT:
				{
					int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						setVariantClassPointers( static_cast<hkVariant*>(maccess.asRaw()), nelem, classFromObject );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_ARRAY:
				case hkClassMember::TYPE_SIMPLEARRAY:
				{
					if( member.getSubType() == hkClassMember::TYPE_VARIANT )
					{
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
							setVariantClassPointers( static_cast<hkVariant*>(array.data), array.size, classFromObject );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
					}
					else if( member.getSubType() == hkClassMember::TYPE_STRUCT )
					{
						HK_ASSERT(0x556380a8, member.hasClass());
						void* obj = pointer;
						int objCount = numObjs;
						while( --objCount >= 0 )
						{
							hkClassMemberAccessor maccess(obj, &member);
							hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
							retrackVariantAndHomogeneousClasses( array.data, member.getStructClass(), classFromObject, array.size );
							obj = hkAddByteOffset(obj, klass.getObjectSize());
						}
					}
					break;
				}
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						hkClassMemberAccessor::HomogeneousArray& homogeneousArray = maccess.asHomogeneousArray();
						if( homogeneousArray.klass )
						{
							objectPointedBy(homogeneousArray.klass, &homogeneousArray.klass);
							retrackVariantAndHomogeneousClasses( homogeneousArray.data, *homogeneousArray.klass, classFromObject, homogeneousArray.size );
						}
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				case hkClassMember::TYPE_STRUCT:
				{
					HK_ASSERT(0x230b134a, member.hasClass());
					int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						retrackVariantAndHomogeneousClasses( maccess.asRaw(), member.getStructClass(), classFromObject, nelem );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
					break;
				}
				default:
				{
					// skip over all other types
				}
			}
		}
	}

	public:

		hkXmlPackfileUpdateTracker( hkPackfileData* data, hkRefPtr<hkChainedClassNameRegistry>& packfileClassRegistry )
			: hkPackfileObjectUpdateTracker(data), m_dirty(false), m_packfileClassRegistry(packfileClassRegistry)
		{
		}

		inline hkBool32 isDirty()
		{
			return m_dirty;
		}

		inline void setDirty()
		{
			m_dirty = true;
			m_packfileClassRegistry = HK_NULL;
		}

		virtual void replaceObject( void* oldObject, void* newObject, const hkClass* newClass )
		{
			setDirty();
			hkPackfileObjectUpdateTracker::replaceObject(oldObject, newObject, newClass);
		}

		virtual void removeFinish( void* oldObject )
		{
			setDirty();
			hkPackfileObjectUpdateTracker::removeFinish(oldObject);
		}

		inline int addPendingValue( void* address, int oldIndex )
		{
			return m_pointers.addPendingValue( address, oldIndex );
		}

		inline void realizePendingPointer( void* newObject, int startIndex )
		{
			int refIndex = startIndex;
			while( refIndex != -1 )
			{
				void* ptrToNew = m_pointers.getValue(refIndex);
				*reinterpret_cast<void**>(ptrToNew) = newObject;
				refIndex = m_pointers.getNextIndex(refIndex);
			}
			m_pointers.realizePendingKey( newObject, startIndex );
		}

		inline void fixAndTrackVariantAndHomogeneousClasses(hkArray<hkVariant>& objects, const ClassFromAddressMap& classFromObject)
		{
			for( int i = 0; i < objects.getSize(); ++i )
			{
				HK_ASSERT(0x2c5efa3a, objects[i].m_class);
				retrackVariantAndHomogeneousClasses(objects[i].m_object, *objects[i].m_class, classFromObject, 1);
			}
		}

		hkBool32 m_dirty;
		hkRefPtr<hkChainedClassNameRegistry>& m_packfileClassRegistry;
};

hkPackfileData* hkXmlPackfileReader::getPackfileData() const
{
	HK_ASSERT2(0x24e3df46, isVersionUpToDate(),
		"Make sure the versioning is completed before accessing the packfile data.");

	if( const hkClassNameRegistry* registry = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
	{
		useClassesFromRegistry(*registry);
	}
	m_data->setPackfileClassNameRegistry(getClassNameRegistry());

	if( !m_data->finishedObjects() )
	{
		for( int loadedIndex = 0; loadedIndex < m_loadedObjects.getSize(); ++loadedIndex )
		{
			hkVariant& loadedObj = m_loadedObjects[loadedIndex];
			if( const char* finishName = m_tracker->m_finish.getWithDefault(loadedObj.m_object, HK_NULL) )
			{
				HK_ASSERT(0x4c45f944, hkString::strCmp(finishName, loadedObj.m_class->getName()) == 0 );
				m_data->trackObject(loadedObj.m_object, loadedObj.m_class->getName());
			}
		}
		m_data->setContentsWithName(m_tracker->m_topLevelObject, getContentsClassName());
	}
	return m_data;
}

hkArray<hkVariant>& hkXmlPackfileReader::getLoadedObjects() const
{
	return m_loadedObjects;
}

hkVariant hkXmlPackfileReader::getTopLevelObject() const
{
	HK_ASSERT(0x7df3a55e, m_tracker);
	hkVariant topLevelObjectVariant = { m_tracker->m_topLevelObject, HK_NULL };
	if( const hkClassNameRegistry* registry = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
	{
		useClassesFromRegistry(*registry);
	}
	topLevelObjectVariant.m_class = getClassNameRegistry()->getClassByName(getContentsClassName());
	return topLevelObjectVariant;
}

hkObjectUpdateTracker& hkXmlPackfileReader::getUpdateTracker() const
{
	return *m_tracker;
}

hkXmlPackfileReader::hkXmlPackfileReader()
{
	// start reading into data section unless otherwise specified
	m_knownSections.pushBack( "__data__" );
	m_sectionTagToIndex.insert( m_knownSections.back(), 0 );
	m_data = new AllocatedData();
	const char* lastVer = hkVersionUtil::getDeprecatedCurrentVersion();
	m_data->setPackfileClassNameRegistry( hkVersionRegistry::getInstance().getClassNameRegistry(lastVer) );
	m_tracker = new hkXmlPackfileUpdateTracker(m_data, m_packfileClassRegistry);
}

hkXmlPackfileReader::~hkXmlPackfileReader()
{
	delete m_tracker;
	m_data->removeReference();
}

hkResult hkXmlPackfileReader::readHeader(hkStreamReader* stream, hkSerializeDeprecated::XmlPackfileHeader& out)
{
	hkPatchClassInstanceXmlParser parser;
	hkStringMap<void*> nameToObject; // map of string id to object address
	hkXmlObjectReader reader(&parser, &nameToObject);
	hkXmlParser::Node* node;
	
	if(parser.nextNode(&node, stream) == HK_FAILURE)
	{
		return HK_FAILURE;
	}

	hkXmlParser::StartElement* startElement = node->asStart();
	if(startElement == HK_NULL)
	{
		delete node;
		return HK_FAILURE;
	}

	if(startElement->name != "hkpackfile")
	{
		delete node;
		return HK_FAILURE;
	}

	if(const char* top = startElement->getAttribute("toplevelobject", HK_NULL))
	{
		out.m_topLevelObject = top;
	}

	const char* versionString = HK_NULL;
	if((versionString = startElement->getAttribute("classversion", HK_NULL)) != HK_NULL)
	{
		out.m_classVersion = hkString::atoi(versionString);
		switch(out.m_classVersion)
		{
		case 1:
			{
				out.m_contentsVersion = "Havok-3.0.0";
				break;
			}
		case 2:
		case 3:
		case 4:
		case 5:
		case 6:
		case 7:
		case 8:
		case 9:
		case 10:
		case 11:
			{
				const char* sdkVersionString = startElement->getAttribute("contentsversion", HK_NULL);
				out.m_contentsVersion = ( sdkVersionString != HK_NULL) ? sdkVersionString : "Havok-3.1.0";

				if(out.m_classVersion < 10)
				{
					break;
				}

				if(const char* maxpredicatesStr = startElement->getAttribute("maxpredicate", HK_NULL))
				{
					
				}
				else
				{
					delete node;
					return HK_FAILURE;
				}

				if(const char* predicatesStr = startElement->getAttribute("predicates", HK_NULL))
				{
					
				}
				else
				{
					delete node;
					return HK_FAILURE;
				}

				break;
			}
		default:
			{
				HK_ASSERT3(0x256758e6, 0, "Unsupported metadata version " << out.m_classVersion);
				delete node;
				return HK_FAILURE;
			}
		}
	}
	else
	{
		out.m_contentsVersion = "Havok-3.0.0";
	}

	delete node;
	return HK_SUCCESS;
}

const hkClass* hkXmlPackfileReader::getClassByName(
		const char* className,
		hkStringMap<hkClass*>& partiallyLoadedClasses,
		hkPointerMap<const hkClass*, int>& offsetsRecomputed,
		int classVersion,
		const char* contentsVersion ) const
{
	// loaded and computed already, first
	const hkClass* klass = m_packfileClassRegistry->getClassByName(className);
	if( !klass )
	{
		// currently missing, but may be loaded
		hkStringMap<hkClass*>::Iterator it = partiallyLoadedClasses.findKey( className );
		if( partiallyLoadedClasses.isValid(it) ) // maybe recompute offsets from loaded instance
		{
			hkClass* c = partiallyLoadedClasses.getValue(it);
			partiallyLoadedClasses.remove(it);

			hkPackfileReader::updateMetaDataInplace( c, classVersion, contentsVersion, m_updateFlagFromClass );

			hkStructureLayout layout;
			layout.computeMemberOffsetsInplace(*c, offsetsRecomputed);
			fixClassEnumClassPointers(*c);
			m_packfileClassRegistry->registerClass(c);
			return c;
		}
	}
	return klass;
}

void hkXmlPackfileReader::handleInterObjectReferences(
	const char* newObjectName, // newly created object name
	void* newObject, // newly created object
	const hkRelocationInfo& reloc,
	const hkStringMap<void*>& nameToObject,
	hkStringMap<int>& unresolvedReferences )
{
	// fixup all pending references to this new object
	{
		hkStringMap<int>::Iterator iter = unresolvedReferences.findKey(newObjectName);
		if( unresolvedReferences.isValid(iter) )
		{
			int refIndex = unresolvedReferences.getValue(iter);
			m_tracker->realizePendingPointer( newObject, refIndex );
			unresolvedReferences.remove(iter);
		}
	}

	// remember any vtable fixups
	{
		if( reloc.m_finish.getSize() )
		{
			//HK_ASSERT(0x5519c131, reloc.m_finish.getSize() == 1 );
			const hkRelocationInfo::Finish& virt = reloc.m_finish[0];
			void* addr = static_cast<char*>(newObject) + virt.m_fromOffset;
			m_tracker->addFinish( addr, virt.m_className );
		}
	}
	
	// fix up any intra-file references in this object
	{
		for( int i = 0; i < reloc.m_imports.getSize(); ++i )
		{
			const hkRelocationInfo::Import& ext = reloc.m_imports[i];

			void* dest = HK_NULL;
			if( nameToObject.get(ext.m_identifier, &dest) == HK_SUCCESS )
			{
				// The external can be resolved immediately
				void* source = static_cast<char*>(newObject) + ext.m_fromOffset;
				
				// Add ourself to the list of objects pointing to dest
				m_tracker->objectPointedBy( dest, source );
			}
			else // pointed object not found yet, add it to pending list
			{
				int oldIndex = unresolvedReferences.getWithDefault(ext.m_identifier, -1);
				int newIndex = m_tracker->addPendingValue( static_cast<char*>(newObject) + ext.m_fromOffset, oldIndex );
				unresolvedReferences.insert( ext.m_identifier, newIndex );
			}
		}
	}
}

inline static hkResult readPackfileAttributes(hkXmlParser::StartElement* startElement, int& classVersion, char**contentsVersion, char**topLevelObjectName)
{
	const char* versionString;
	
	*topLevelObjectName = HK_NULL;
	if( const char* top = startElement->getAttribute("toplevelobject", HK_NULL) )
	{
		*topLevelObjectName = hkString::strDup(top);
	}
	if( (versionString = startElement->getAttribute("classversion", HK_NULL)) != HK_NULL )
	{
		classVersion = hkString::atoi(versionString);
		switch( classVersion )
		{
			case 1:
			{
				*contentsVersion = hkString::strDup("Havok-3.0.0");
				break;
			}
			case 2:
			case 3:
			case 4:
			case 5:
			case 6:
			case 7:
			case 8:
			case 9:
			case 10:
			case 11:
			{
				const char* sdkVersionString = startElement->getAttribute("contentsversion", HK_NULL);
				*contentsVersion = ( sdkVersionString != HK_NULL)
					? hkString::strDup(sdkVersionString)
					: hkString::strDup("Havok-3.1.0");
				break;
			}
			default:
			{
				HK_ASSERT3(0x256758e6, 0, "Unsupported metadata version " << classVersion);
				return HK_FAILURE;
			}
		}
	}
	else
	{
		*contentsVersion = hkString::strDup("Havok-3.0.0");
	}
	return HK_SUCCESS;
}

hkResult hkXmlPackfileReader::loadEntireFile( hkStreamReader* rawStream )
{
	return loadEntireFileWithRegistry(rawStream, HK_NULL);
}

hkResult hkXmlPackfileReader::loadEntireFileWithRegistry( hkStreamReader* rawStream, const hkClassNameRegistry* originalReg )
{
	HK_ASSERT2( 0x6088f580, m_data->isDirty() == false, "You must use new reader for each time you load a packfile.");
	TRACE_FUNC_ENTER((">> loadEntireFileWithRegistry:"));
	hkLocalArray<char> buf( 0x4000 ); // 16k default buffer size.

	bool classRegistryAutoDetected = false;
	int currentSectionIndex = 0;
	hkStreamReader* stream = rawStream;
	hkPatchClassInstanceXmlParser parser;
	hkRelocationInfo reloc;
	hkXmlParser::Node* node;
	hkStorageStringMap<int> stringPool;

	hkPointerMap<const hkClass*,int> offsetsRecomputed; // retargeted classes
	hkStringMap<hkClass*> partiallyLoadedClasses; // classes partially loaded
	hkStringMap<int> unresolvedReferences; // map of string id to reference chain index
	hkStringMap<void*> nameToObject; // map of string id to object address
	hkXmlObjectReader reader(&parser, &nameToObject);
	ClassFromAddressMap classFromObject;
	char* topLevelObjectName = HK_NULL;

	// make sure the class registry is available, initially next registry may be HK_NULL (contents version is not available)
	getClassNameRegistry();
	// register internal classes, we do not know the contents version yet
	m_packfileClassRegistry->registerClass( &hkClassVersion1PaddedClass );
	m_packfileClassRegistry->registerClass( &hkClassEnumVersion1PaddedClass );

	if( originalReg )
	{
		// copy classes from provided registry
		hkArray<const hkClass*> classes;
		originalReg->getClasses(classes);
		for( int i = 0; i < classes.getSize(); ++i )
		{
			HK_ASSERT(0x1044b08d, classes[i]);
			m_packfileClassRegistry->registerClass(classes[i]);
		}
	}

	nameToObject.insert("null", HK_NULL);
	int classVersion = 1;
#	if defined(HK_DEBUG)
		// hkClass signature fix. HVK-3680
		bool hasCorrectSig = false;
#	endif

	while( parser.nextNode(&node, stream) == HK_SUCCESS )
	{
		if( hkXmlParser::StartElement* startElement = node->asStart() )
		{
			if( startElement->name == "hkobject")
			{
				// peek at the type of the object
				const hkClass* klass = HK_NULL;
				const char* objectName = HK_NULL;
				const char* exportName = HK_NULL;
				{
					const char* className = startElement->getAttribute("class", HK_NULL );
					HK_ASSERT2(0x6087f47f,className != HK_NULL, "Found XML element without a 'class' attribute." );
					klass = getClassByName( className, partiallyLoadedClasses, offsetsRecomputed, classVersion, getContentsVersion() );

					objectName = startElement->getAttribute("name", HK_NULL );
					HK_ASSERT2(0x6088e48e,objectName!= HK_NULL, "Found XML element with no 'name' attribute." );
					HK_ASSERT3(0x6088f48f,klass!= HK_NULL, "Found '" << objectName << "' with an unregistered type '" << className << "'");

					if( klass == HK_NULL )
					{
						return HK_FAILURE;
					}

					objectName = stringPool.insert(objectName, 0);

					//TRACE(("OBJ %s %s\n", objectName, className));
					exportName = startElement->getAttribute("export", HK_NULL );
					if( exportName ) // keep this for later
					{
						char* name = hkString::strDup(exportName);
						exportName = name;
						m_data->addAllocation(name);
					}
#					if defined(HK_DEBUG)
						if( hasCorrectSig /* hkClass signature fix. HVK-3680 */
							&& xmlPackFileReader_IsPublicClassObject(className) )
						{
							const char* signature = startElement->getAttribute("signature", HK_NULL );
							if( signature )
							{
								hkUint32 lsig = hkString::atoi(signature);
								hkUint32 msig = klass->getSignature();
								HK_ASSERT2( 0x26fd9749, msig == lsig, "metadata signature mismatch" );
							}
						}
#					endif
				}

				parser.putBack(node);
				node = HK_NULL;

				// read the object into a temporary buffer
				buf.clear();
				if( reader.readObject(stream, buf, *klass, reloc ) != HK_SUCCESS )
				{
					return HK_FAILURE;
				}
				HK_ASSERT2(0x2dbfd9dd, reloc.m_global.getSize() == 0, "Object has read global relocations during XML read." );
				void* object = HK_NULL;
				// make sure that we reuse already registered hkClass instances
				if( hkString::strCmp(klass->getName(), "hkClass") == 0 )
				{
					reloc.applyLocalAndGlobal(buf.begin()); // actually only local
					const hkClass* k = getClassByName( reinterpret_cast<hkClass*>(buf.begin())->getName(), partiallyLoadedClasses, offsetsRecomputed, classVersion, getContentsVersion() );
					if( k )
					{
						object = const_cast<hkClass*>(k);
						reloc.clear();
					}
				}

				// check if not using existing object
				if( object == HK_NULL )
				{
					// copy it into final location and apply local fixups
					object = hkAllocate<char>(buf.getSize(), HK_MEMORY_CLASS_EXPORT);
					hkString::memCpy( object, buf.begin(), buf.getSize() );
					reloc.applyLocalAndGlobal(object); // actually only local
					m_data->addAllocation(object);
				}

				if( exportName )
				{
					m_data->addExport( exportName, object );
				}

				// object is now created
				nameToObject.insert( objectName, object );
				handleInterObjectReferences( objectName, object, reloc, nameToObject, unresolvedReferences );

				if( hkString::strCmp(klass->getName(), "hkClass") == 0 )
				{
					hkClass* k = static_cast<hkClass*>(object);
					TRACE("Load class " << k->getName() << " (0x" << k << ").");
					partiallyLoadedClasses.insert(k->getName(), k);
				}
				else if ( xmlPackFileReader_IsPublicClassObject(klass->getName()) )
				{
					hkVariant& v = m_loadedObjects.expandOne();
					v.m_class = klass;
					v.m_object = object;
					classFromObject.insert(object, klass);
					if( currentSectionIndex == 0 && m_tracker->m_topLevelObject == HK_NULL
						&& ((topLevelObjectName == HK_NULL && classVersion < 6) || hkString::strCmp(topLevelObjectName, objectName) == 0) )
					{
						// first data object is contents, if classVersion < 6
						m_tracker->setTopLevelObject(v.m_object, v.m_class->getName());
					}
				}

				reloc.clear();
			}
			else if( startElement->name == "hksection")
			{
				const char* sectName = startElement->getAttribute("name", "__data__");
				HK_ASSERT( 0x470b6e11, getContentsVersion() != HK_NULL);
				if( ::mayContainWrongMetadata(getContentsVersion()) )
				{
					parser.allowClassInstancesOnly( hkString::strCmp(sectName, "__types__") == 0 ); // true when about to process metadata
				}
				if( m_sectionTagToIndex.get(sectName, &currentSectionIndex) != HK_SUCCESS )
				{
					currentSectionIndex = m_knownSections.getSize();
					char* s = hkString::strDup(sectName);
					m_data->addAllocation( s );
					m_knownSections.pushBack( s );
					m_sectionTagToIndex.insert( m_knownSections.back(), currentSectionIndex );
				}
			}
			else if( startElement->name == "hkpackfile")
			{
				char* contentsVersion = HK_NULL;
				readPackfileAttributes(startElement, classVersion, &contentsVersion, &topLevelObjectName);
				setContentsVersion( contentsVersion );
				hkDeallocate(contentsVersion); contentsVersion = HK_NULL;
				if( classVersion >= 6 )
				{
					HK_ASSERT3(0x5c4378fa, topLevelObjectName, "XML file version " << classVersion << " must have 'toplevelobject' attribute.");
				}

				if( classVersion >= 8 )
				{
					m_packfileClassRegistry->registerClass( &hkClassClass );
					m_packfileClassRegistry->registerClass( &hkClassEnumClass );
				}
				else if( classVersion >= 5 )
				{
					m_packfileClassRegistry->registerClass( &hkClassVersion3Class );
					m_packfileClassRegistry->registerClass( &hkClassEnumClass );
				}
				else if( classVersion >= 2 )
				{
					m_packfileClassRegistry->registerClass( &hkClassVersion2PaddedClass );
				}
				if( getContentsVersion() )
				{
					// contents version is available, set next registry to a compiled in version, if any.
					if( const hkClassNameRegistry* nextRegistry = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
					{
						HK_ASSERT2( 0x470b6e07, classRegistryAutoDetected == false, "Duplicate <packfile> element found");
						m_packfileClassRegistry->setNextRegistry(nextRegistry);
						classRegistryAutoDetected = true;
					}
				}
#				if defined(HK_DEBUG)
					// hkClass signature fix. HVK-3680
					hasCorrectSig = xmlPackFileReader_containsCorrectSignatures(getContentsVersion());
#				endif
			}
			else
			{
				HK_WARN( 0x5c4378f9, "Unhandled xml tag " << startElement->name);
				return HK_FAILURE;
			}
		}
		else if( hkXmlParser::EndElement* end = node->asEnd() )
		{
			HK_ASSERT3(0x68293525, end->name == "hksection" || end->name == "hkpackfile",
				"Mismatched xml end tag " << end->name);
		}
		else if (hkXmlParser::Characters* chars = node->asCharacters() )
		{
			chars->canonicalize();
			if (chars->text.getLength() > 0)
			{
				// Unidentified text outside of tags
				return HK_FAILURE;
			}
		}
		else
		{
			HK_ERROR(0x5ef4e5a3, "Unhandled tag in XML");
		}
		delete node;
	}

	hkDeallocate(topLevelObjectName);

	hkBool unresolvedLocals = false;
	for( hkStringMap<int>::Iterator it = unresolvedReferences.getIterator(); unresolvedReferences.isValid(it); it = unresolvedReferences.getNext(it) )
	{
		const char* name = unresolvedReferences.getKey(it);
		if( name[0] == '#' )
		{
			// #foo identifiers are local to file
			HK_WARN(0x52a902e9, "undefined reference to '" << name << "'");
			unresolvedLocals = true;
		}
		else if( name[0] == '@' )
		{
			char* importName = hkString::strDup(name+1);
			name = importName;
			m_data->addAllocation(importName);

			int refIndex = unresolvedReferences.getValue(it);
			while( refIndex != -1 )
			{
				void* loc = m_tracker->m_pointers.getValue(refIndex);
				m_data->addImport( name, reinterpret_cast<void**>(loc) );
				refIndex = m_tracker->m_pointers.getNextIndex(refIndex);
			}
		}
		else
		{
			HK_ASSERT3( 0xafdee3b9, 0, "Unrecognised symbol name '" << name << "'" );
		}
	}
	HK_ASSERT2(0x371105b7, unresolvedLocals == false, "File has unresolved references");

	// all references are solved, fix and track classes now
	m_tracker->fixAndTrackVariantAndHomogeneousClasses(m_loadedObjects, classFromObject);

	// update registry with partially loaded classes now, all references are resolved
	for( hkStringMap<hkClass*>::Iterator iter = partiallyLoadedClasses.getIterator();
		partiallyLoadedClasses.isValid(iter); iter = partiallyLoadedClasses.getNext(iter) )
	{
		const hkClass* klass = partiallyLoadedClasses.getValue(iter);
		HK_ASSERT(0x675b72fe, klass);
		if( m_packfileClassRegistry->getClassByName(klass->getName()) == HK_NULL )
		{
			m_packfileClassRegistry->registerClass(klass);
		}
	}

	TRACE_FUNC_EXIT(("<< loadEntireFileWithRegistry."));
	return HK_SUCCESS;
}

void* hkXmlPackfileReader::getContentsWithRegistry(
		const char* expectedClassName,
		const hkTypeInfoRegistry* finishRegistry )
{
	if( finishRegistry != HK_NULL )
	{
		warnIfNotUpToDate();
		getPackfileData();  // make sure packfile data tracks objects
	}
	else
	{
		if( !getContentsVersion() )
		{
			return HK_NULL;
		}
		if( const hkClassNameRegistry* classReg = getGlobalRegistryForPackfileVersion(getContentsVersion()) )
		{
			useClassesFromRegistry(*classReg);
		}
		m_data->setPackfileClassNameRegistry(m_packfileClassRegistry);
	}
	return m_data->getContentsPointer(expectedClassName, finishRegistry);
}

const char* hkXmlPackfileReader::getContentsClassName() const
{
	return m_tracker->getTopLevelClassName();
}

const hkClassNameRegistry* hkXmlPackfileReader::getClassNameRegistry() const
{
	if( m_packfileClassRegistry )
	{
		return m_packfileClassRegistry;
	}
	TRACE_FUNC_ENTER(">> getClassNameRegistry:");
	const hkClassNameRegistry* nextRegistry = getGlobalRegistryForPackfileVersion(getContentsVersion());
	m_packfileClassRegistry = new hkChainedClassNameRegistry(HK_NULL);
	m_packfileClassRegistry->removeReference(); // hkRefPtr takes responsibility

	hkXmlPackfileUpdateTracker& tracker = static_cast<hkXmlPackfileUpdateTracker&>(getUpdateTracker());
	if( tracker.isDirty() )
	{
		hkArray<hkVariant>& loadedObject = m_loadedObjects;
		for( int i = 0; i < loadedObject.getSize(); ++i )
		{
			hkClass* klass = const_cast<hkClass*>(loadedObject[i].m_class);
			HK_ASSERT(0x1775fdc0, klass);
			validateAndRegisterClasses(*m_packfileClassRegistry, klass);
			findAndRegisterHomogeneousClassesIn(*m_packfileClassRegistry, loadedObject[i].m_object, *klass, 1);
		}
	}
	m_packfileClassRegistry->setNextRegistry(nextRegistry);
	TRACE_FUNC_EXIT("<< getClassNameRegistry.");

	return m_packfileClassRegistry;
}

void hkXmlPackfileReader::useClassesFromRegistry(const hkClassNameRegistry& registry) const
{
	TRACE_FUNC_ENTER(">> useClassesFromRegistry:");
	// make sure we track packfile object and class locations
	hkRefPtr<hkChainedClassNameRegistry> classReg = static_cast<hkChainedClassNameRegistry*>(const_cast<hkClassNameRegistry*>(getClassNameRegistry())); // read metadata only, if any
	hkArray<const hkClass*> classes;
	classReg->getClasses(classes);
	for( int i = 0; i < classes.getSize(); ++i )
	{
		const hkClass* klass = classes[i];
		const hkClass* registryClass = registry.getClassByName(klass->getName());
		if( registryClass )
		{
			if( registryClass != klass )
			{
				// special case for hkClassMember and hkClass metadata loaded from old files.
				HK_ON_DEBUG(if((hkString::strCmp(klass->getName(), "hkClassMember") != 0 || klass->getSignature() != 0xb0efa719) && (hkString::strCmp(klass->getName(), "hkClass") != 0 || klass->getSignature() != 0x33d42383)))
					HK_ASSERT3(0x741cc3a5, klass->getSignature() == registryClass->getSignature(), "Signature mismatch " << klass->getName() << " (0x" << klass << ") - 0x" << (void*)((hkUlong)klass->getSignature()) << ", 0x" << (void*)((hkUlong)registryClass->getSignature()));
				// replace class with version from registry
				classReg->registerClass(registryClass);
				TRACE("Class " << klass->getName() << " is used from the provided registry.");
			}
		}
		else
		{
			HK_WARN(0x7bfbc4c0, "Class " << klass->getName() << " is not found in the provided registry.");
		}
	}
	TRACE_FUNC_EXIT("<< useClassesFromRegistry.");
}

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
