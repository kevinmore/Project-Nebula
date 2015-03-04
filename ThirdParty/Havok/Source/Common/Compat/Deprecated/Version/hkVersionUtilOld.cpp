/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/hkSerialize.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Reflection/hkInternalClassMember.h>
#include <Common/Base/System/Io/Writer/Buffered/hkBufferedStreamWriter.h>
#include <Common/Base/Types/hkTypedUnion.h>
#include <Common/Serialize/Copier/hkObjectCopier.h>
#include <Common/Compat/Deprecated/Packfile/hkPackfileReader.h>
#include <Common/Serialize/Serialize/hkRelocationInfo.h>
#include <Common/Compat/Deprecated/Util/hkRenamedClassNameRegistry.h>
#include <Common/Serialize/Util/hkSerializeLog.h>
#include <Common/Compat/Deprecated/Version/hkObjectUpdateTracker.h>

extern const hkClass hkClassMemberClass;
namespace
{
#	define _NEW_LINE_ "\r\n"

	static const char s_classesHeaderBegin[] = "#ifndef %s" _NEW_LINE_ "#define %s" _NEW_LINE_;
	static const char s_classesHeaderEnd[] = "#endif // %s";
	static const char s_commonHeaders[] =
//		"#include <Common/Base/Reflection/hkClass.h>"   _NEW_LINE_
//		"#include <Common/Base/Reflection/hkInternalClassMember.h>"   _NEW_LINE_ _NEW_LINE_
		"#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>" _NEW_LINE_ _NEW_LINE_;
	static const char s_namespaceOpen[] = "namespace hkHavok%sClasses" _NEW_LINE_ "{" _NEW_LINE_;
	static const char s_commonForwardDeclaration[] =
		"\textern const char VersionString[];" _NEW_LINE_
		"\textern const int ClassVersion;" _NEW_LINE_
		"\textern const hkStaticClassNameRegistry %s;"   _NEW_LINE_; // registryVariableName
	static const char s_namespaceClose[] = "} // namespace hkHavok%sClasses" _NEW_LINE_;
	static const char s_externClass[] = "\textern hkClass %sClass; /* 0x%p */" _NEW_LINE_;

	// default
	static const char s_defaultsNamespaceOpen[] ="\tnamespace" _NEW_LINE_ "\t{" _NEW_LINE_;
	static const char s_defaultStructDefinition[] =
		"\t\tstruct %s_DefaultStruct" _NEW_LINE_ "\t\t{" _NEW_LINE_ /* class name */
		"\t\t\tint s_defaultOffsets[%d];" _NEW_LINE_ /* number of declared members */
		"\t\t\ttypedef hkInt8 _hkBool;" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkVector4[4];" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkQuaternion[4];" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkMatrix3[12];" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkRotation[12];" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkQsTransform[12];" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkMatrix4[16];" _NEW_LINE_ /**/
		"\t\t\ttypedef hkReal _hkTransform[16];" _NEW_LINE_ /**/
		"%s" /* members with defaults, e.g. hkReal m_hierarchyGain; */
		"\t\t};"   _NEW_LINE_;
	static const char s_defaultsStructMember[] = "\t\t\t%s m_%s;" _NEW_LINE_; /* class member type, class member name */
	static const char s_defaultsDefinition[] =
		"\t\tconst %s_DefaultStruct %s_Default ="   _NEW_LINE_"\t\t{"   _NEW_LINE_ /* class name, class name */
		"\t\t\t{%s},"   _NEW_LINE_ /* list of offsets for members with default values, e.g. HK_OFFSET_OF(hkaKeyFrameHierarchyUtilityControlData_DefaultStruct,m_hierarchyGain) */
		"\t\t\t%s"   _NEW_LINE_ /* list of defaults */
		"\t\t};"   _NEW_LINE_;
	static const char s_defaultsMemDefinition[] = "HK_OFFSET_OF(%s_DefaultStruct,m_%s)"; /* class name, class member name */
	static const char s_defaultsNamespaceClose[] = "\t}"   _NEW_LINE_;

	// enum items
	static const char s_enumItemsStart[] = "\tstatic const hkInternalClassEnumItem %sEnumItems[] ="   _NEW_LINE_"\t{"   _NEW_LINE_; /* className + enumName */
	static const char s_enumItemDefinition[] = "\t\t{%d, \"%s\"},"   _NEW_LINE_; /* value, name */
	static const char s_enumItemsEnd[] = "\t};"   _NEW_LINE_;
	// enums
	static const char s_enumsStart[] = "\tstatic const hkInternalClassEnum %sEnums[] ="   _NEW_LINE_"\t{"   _NEW_LINE_; /* className */
	static const char s_enumDefinition[] = "\t\t{\"%s\", %sEnumItems, %d, HK_NULL, %d },"   _NEW_LINE_; /* enum name, className + enum name, numItems, attributes, flags */
	static const char s_enumsEnd[] = "\t};"   _NEW_LINE_;
	static const char s_enumExtern[] = "\textern const hkClassEnum* %sEnum = reinterpret_cast<const hkClassEnum*>(&%sEnums[%d]);"   _NEW_LINE_; /* enumName, className, enum index */
	static const char s_enumForward[] = "\textern const hkClassEnum* %sEnum;"   _NEW_LINE_; /* enumName */
	// class members
	static const char s_membersStart[] = "\tstatic hkInternalClassMember %sClass_Members[] ="   _NEW_LINE_"\t{"   _NEW_LINE_; /* class name */
	static const char s_memberDefinition[] =
		"\t\t{ \"%s\", " /* memberName */
		"%s, " /* class pointer */
		"%s, " /* enum pointer */
		"hkClassMember::%s, " /* havokType, e.g. hkClassMember::TYPE_VECTOR4 */
		"hkClassMember::%s, " /* havokSubType, e.g. hkClassMember::TYPE_VOID */
		"%d, " /* cArraySize */
		"%d, " /* flags */
		"0, " /* offset */
		"HK_NULL },"   _NEW_LINE_; /* attributes */
	static const char s_membersEnd[] = "\t};"   _NEW_LINE_;
	// class
	static const char s_classDefinition[] =
		"\thkClass %sClass("   _NEW_LINE_ /* hkClass variable name */
		"\t\t\"%s\","   _NEW_LINE_ /* className */
		"\t\t%s,"   _NEW_LINE_ /* parentClass */
		"\t\t0,"   _NEW_LINE_ /* objectSizeInBytes */
		"\t\t%s,"   _NEW_LINE_ /* implementedInterfaces */
		"\t\t%d,"   _NEW_LINE_ /* numImplementedInterfaces */
		"\t\t%s,"   _NEW_LINE_ /* declaredEnums */
		"\t\t%d,"   _NEW_LINE_ /* numDeclaredEnums */
		"\t\t%s,"   _NEW_LINE_ /* declaredMembers */
		"\t\t%s,"   _NEW_LINE_ /* numDeclaredMembers */
		"\t\t%s"   _NEW_LINE_"\t);"   _NEW_LINE_; /* defaults - variable name */

	static const char s_classStaticRegistry[] =
		"\tstatic hkClass* const Classes[] ="   _NEW_LINE_"\t{"   _NEW_LINE_"%s\t\tHK_NULL"   _NEW_LINE_"\t};"   _NEW_LINE_ _NEW_LINE_ // class list, e.g \t\t&name,\r\n\t\t&name2,\r\n
		"\tconst hkStaticClassNameRegistry %s(Classes, ClassVersion, VersionString);"   _NEW_LINE_; // registryVariableName

	static inline void generateEnum(const char* completeEnumName, const hkClassEnum& classEnum, hkStringBuf& buffer)
	{
		buffer.appendPrintf(s_enumItemsStart, completeEnumName);
		for( int item = 0; item < classEnum.getNumItems(); ++item )
		{
			const hkClassEnum::Item& enumItem = classEnum.getItem(item);
			buffer.appendPrintf(s_enumItemDefinition, enumItem.getValue(), enumItem.getName());
		}
		buffer += s_enumItemsEnd;
	}

	template <typename T>
	const T& lookupMember(const void* start)
	{
		return *reinterpret_cast<const T*>( start );
	}

	static void generateMemberDefaultValue(const hkTypedUnion& value, hkClassMember::Type type, hkClassMember::Type subType, hkStringBuf& outputString)
	{
		const void* valueAddress = &value.getStorage();
		switch(type)
		{
			case hkClassMember::TYPE_BOOL:
			{
				outputString = (lookupMember<hkBool>(valueAddress) ? "true" : "false");
				break;
			}
			case hkClassMember::TYPE_CHAR:
			{
				outputString.printf("'%c'", lookupMember<char>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_INT8:
			{
				int i = lookupMember<hkInt8>(valueAddress);
				outputString.printf("%i", i);
				break;
			}
			case hkClassMember::TYPE_UINT8:
			{
				outputString.printf("%u", lookupMember<hkUint8>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_INT16:
			{
				outputString.printf("%i", lookupMember<hkInt16>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_UINT16:
			{
				outputString.printf("%u", lookupMember<hkUint16>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_INT32:
			{
				outputString.printf("%i", lookupMember<hkInt32>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_UINT32:
			{
				outputString.printf("%u", lookupMember<hkUint32>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_INT64:
			{
				outputString.printf(HK_PRINTF_FORMAT_INT64, lookupMember<hkInt64>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_UINT64:
			{
				outputString.printf(HK_PRINTF_FORMAT_UINT64, lookupMember<hkUint64>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_ULONG:
			{
				outputString.printf(HK_PRINTF_FORMAT_ULONG, lookupMember<hkUlong>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_REAL:
			{
				outputString.printf("%ff", lookupMember<hkReal>(valueAddress));
				break;
			}
			case hkClassMember::TYPE_HALF:
			{
				outputString.printf("%ff", lookupMember<hkHalf>(valueAddress).getReal());
				break;
			}
			case hkClassMember::TYPE_VECTOR4:
			case hkClassMember::TYPE_QUATERNION:
			{
				const hkReal* r = reinterpret_cast<const hkReal*>( valueAddress );
				outputString.printf("{%ff, %ff, %ff, %ff}", r[0], r[1], r[2], r[3]);
				break;
			}
			case hkClassMember::TYPE_MATRIX3:
			case hkClassMember::TYPE_ROTATION:
			case hkClassMember::TYPE_QSTRANSFORM:
			{
				const hkReal* r = reinterpret_cast<const hkReal*>( valueAddress );
				outputString.printf("{%ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff}",
					r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11]);
				break;
			}
			case hkClassMember::TYPE_MATRIX4:
			case hkClassMember::TYPE_TRANSFORM:
			{
				const hkReal* r = reinterpret_cast<const hkReal*>( valueAddress );
				outputString.printf("{%ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff, %ff}",
					r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]);
				break;
			}
			case hkClassMember::TYPE_ENUM:
			case hkClassMember::TYPE_FLAGS:
			{
				generateMemberDefaultValue(value, subType, hkClassMember::TYPE_VOID, outputString);
				break;
			}
			//TYPE_POINTER
			//TYPE_CSTRING
			//TYPE_STRINGPTR
			//TYPE_FUNCTIONPOINTER
			//TYPE_VARIANT
			//TYPE_ARRAY
			//TYPE_INPLACEARRAY
			//TYPE_STRUCT
			//TYPE_SIMPLEARRAY
			//TYPE_HOMOGENEOUSARRAY
			default:
			{
				HK_ASSERT3(0x10cb67f1, 0, "Unhandled type '" << hkClassMember::getClassMemberTypeProperties(type).m_name << "'.");
			}
		}
	}

	static inline void generateMemberTypeNameForDefaults(hkClassMember::Type type, hkClassMember::Type subType, hkStringBuf& outputString)
	{
		outputString = "";
		if( (type >= hkClassMember::TYPE_BOOL && type <= hkClassMember::TYPE_TRANSFORM)
			|| type == hkClassMember::TYPE_ULONG )
		{
			if( type == hkClassMember::TYPE_BOOL
				|| (type >= hkClassMember::TYPE_VECTOR4 && type <= hkClassMember::TYPE_TRANSFORM) )
			{
				outputString = "_";
			}
			outputString += hkClassMember::getClassMemberTypeProperties(type).m_name;
		}
		else if( type == hkClassMember::TYPE_ENUM || type == hkClassMember::TYPE_FLAGS )
		{
			outputString = hkClassMember::getClassMemberTypeProperties(subType).m_name;
		}
		else
		{
			HK_ASSERT3(0x10cb67f2, false, "Unhandled type '" << hkClassMember::getClassMemberTypeProperties(type).m_name << "'." );
		}
	}

	static inline void generateDefaults(const hkClass& klass, hkArray<int>& memIdxWithDefault, hkStringBuf& outputString)
	{
		HK_ASSERT(0x10cb67f3, memIdxWithDefault.getSize() > 0);
		outputString = s_defaultsNamespaceOpen;
		hkInt32 defaultIdx = 0;
		hkStringBuf membersList;
		// generate list of members with defaults
		for( int i = 0; i < memIdxWithDefault.getSize(); ++i )
		{
			HK_ASSERT(0x10cb67f4, memIdxWithDefault[i] < klass.getNumDeclaredMembers());
			const hkClassMember& mem = klass.getDeclaredMember(memIdxWithDefault[i]);
			hkStringBuf memType;
			generateMemberTypeNameForDefaults(mem.getType(), mem.getSubType(), memType);
			membersList.appendPrintf(s_defaultsStructMember, memType.cString(), mem.getName());
		}
		outputString.appendPrintf(s_defaultStructDefinition, klass.getName(), klass.getNumDeclaredMembers(), membersList.cString());
		hkStringBuf defaultOffsetsList;
		hkStringBuf defaultValuesList;
		for( int i = 0; i < klass.getNumDeclaredMembers(); ++i )
		{
			if( defaultIdx < memIdxWithDefault.getSize() && memIdxWithDefault[defaultIdx] == i )
			{
				const hkClassMember& mem = klass.getDeclaredMember(i);
				// offset
				defaultOffsetsList.appendPrintf(s_defaultsMemDefinition, klass.getName(), mem.getName());
				defaultOffsetsList += ",";
				// value
				hkTypedUnion val;
				HK_ON_DEBUG(hkResult res = ) klass.getDefault(klass.getMemberIndexByName(mem.getName()), val);
				HK_ASSERT(0x10cb67f5, res == HK_SUCCESS);
				hkStringBuf buffer;
				generateMemberDefaultValue(val, mem.getType(), mem.getSubType(), buffer);
				defaultValuesList.appendJoin(buffer, ",");
				defaultIdx++;
			}
			else
			{
				defaultOffsetsList += "-1,";
			}
		}
		outputString.appendPrintf(s_defaultsDefinition, klass.getName(), klass.getName(), defaultOffsetsList.cString(), defaultValuesList.cString());
		outputString += s_defaultsNamespaceClose;
	}

	// COM-629, c-string -> hkStringPtr, hkSimpleArray -> hkArray
	class PackfileObjectCopier: public hkObjectCopier
	{
		public:
			PackfileObjectCopier(const hkStructureLayout& layoutIn, const hkStructureLayout& layoutOut);

		protected:
			virtual hkBool32 areMembersCompatible(const hkClassMember& src, const hkClassMember& dst);
	};

	PackfileObjectCopier::PackfileObjectCopier(const hkStructureLayout& layoutIn, const hkStructureLayout& layoutOut)
		: hkObjectCopier(layoutIn, layoutOut)
	{
	}
	
	// COM-629, c-string -> hkStringPtr, hkSimpleArray -> hkArray
	hkBool32 PackfileObjectCopier::areMembersCompatible(const hkClassMember& src, const hkClassMember& dst)
	{
		bool equalOrCompatibleArrayType = src.getType() == dst.getType() || ( src.getType() == hkClassMember::TYPE_SIMPLEARRAY && dst.getType() == hkClassMember::TYPE_ARRAY );
		bool equalOrCompatibleStringType = src.getType() == hkClassMember::TYPE_CSTRING && dst.getType() == hkClassMember::TYPE_STRINGPTR;
		bool equalOrCompatibleVariantType = src.getType() == hkClassMember::TYPE_VARIANT && dst.getType() == hkClassMember::TYPE_POINTER && dst.getSubType() == hkClassMember::TYPE_STRUCT
											&& dst.getClass() && hkString::strCmp(hkReferencedObjectClass.getName(), dst.getStructClass().getName()) == 0;
		bool equalOrCompatibleStringSubType = src.getSubType() == dst.getSubType() || ( src.getSubType() == hkClassMember::TYPE_CSTRING && dst.getSubType() == hkClassMember::TYPE_STRINGPTR );
		bool equalOrCompatibleVariantSubType = src.getSubType() == hkClassMember::TYPE_VARIANT && dst.getSubType() == hkClassMember::TYPE_POINTER
												&& dst.getClass() && hkString::strCmp(hkReferencedObjectClass.getName(), dst.getStructClass().getName()) == 0;
		return hkObjectCopier::areMembersCompatible(src, dst)
			|| ( equalOrCompatibleStringType )
			|| ( equalOrCompatibleVariantType )
			|| ( equalOrCompatibleArrayType && ( equalOrCompatibleStringSubType || equalOrCompatibleVariantSubType ) );
	}

	template<class T>
	static int getNumElements(T** p)
	{
		int i = 0;
		while( *p != HK_NULL )
		{
			++i;
			++p;
		}
		return i;
	}

	static void* copySomeObjects(
		const hkArrayBase<hkVariant>& objectsIn,
		hkArray<hkVariant>& objectsOut,
		const hkArray<int>::Temp& objectIndices,
		hkObjectUpdateTracker& tracker )
	{
		hkArray<char> buffer;
		hkOstream out(buffer);
		hkRelocationInfo relocations;
		hkPointerMap<const void*, int> overrides;

		for( int objectIndexIndex = 0; objectIndexIndex < objectIndices.getSize(); ++objectIndexIndex )
		{
			int objectIndex = objectIndices[objectIndexIndex];
			const hkVariant& oldVariant = objectsIn[objectIndex];

			PackfileObjectCopier copier( hkStructureLayout::HostLayoutRules, hkStructureLayout::HostLayoutRules );
			const hkClass* newClass = objectsOut[objectIndex].m_class;
			objectsOut[objectIndex].m_object = (void*)static_cast<hkUlong>(buffer.getSize()); // offset - convert to ptr later.

			overrides.insert( oldVariant.m_object, buffer.getSize() );

			copier.copyObject(oldVariant.m_object, *oldVariant.m_class,
				out.getStreamWriter(), *newClass,
				relocations );
		}

		if( buffer.getSize() ) // some versioning was done
		{
			char* versioned = hkAllocateChunk<char>( buffer.getSize(), HK_MEMORY_CLASS_EXPORT );
			tracker.addChunk(versioned, buffer.getSize(), HK_MEMORY_CLASS_EXPORT );
			hkString::memCpy( versioned, buffer.begin(), buffer.getSize() );

			// object pointers were actually stored as offsets because of resizing
			for( int objectIndexIndex = 0; objectIndexIndex < objectIndices.getSize(); ++objectIndexIndex )
			{
				int objectIndex = objectIndices[objectIndexIndex];
				hkVariant& v = objectsOut[objectIndex];
				v.m_object = versioned + hkUlong(v.m_object);
			}

			hkArray<hkRelocationInfo::Local>& local = relocations.m_local;
			for( int localIndex = 0; localIndex < local.getSize(); ++localIndex )
			{
				*(void**)(versioned + local[localIndex].m_fromOffset) = versioned + local[localIndex].m_toOffset;
			}
			hkArray<hkRelocationInfo::Global>& global = relocations.m_global;
			for( int globalIndex = 0; globalIndex < global.getSize(); ++globalIndex )
			{
				void* p = global[globalIndex].m_toAddress;
				void* from = versioned + global[globalIndex].m_fromOffset;
				tracker.objectPointedBy( p, from );
			}
			return versioned;
		}
		return HK_NULL;
	}
}

CollectClassDefinitions::CollectClassDefinitions(const hkArray<const hkClass*>& originalClassList,
												 hkPointerMap<const hkClassEnum*, char*>& enumNameFromPointer,
												 hkStringMap<hkBool32>& enumDoneFlagFromName)
:	m_originalClassList(originalClassList), 
	m_enumNameFromPointer(enumNameFromPointer), 
	m_enumDoneFlagFromName(enumDoneFlagFromName)
{
	m_classMemberEnumType = hkClassMemberClass.getEnumByName("Type");
}

const char* CollectClassDefinitions::getClassExternList() const
{
	return m_classExternList;
}

const char* CollectClassDefinitions::getClassDefinitionList() const
{
	return m_classDefinitionList;
}

void CollectClassDefinitions::defineClassClass(const hkClass& klass)
{
	if( m_doneClasses.hasKey(klass.getName()) )
	{
		return;
	}

	m_doneClasses.insert(klass.getName(), true);

	if( klass.getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
	{
		if( klass.getParent() )
		{
			defineClassClass(*klass.getParent());
		}
		return;
	}

	hkStringBuf buffer;
	buffer.printf(s_externClass, klass.getName(), (void*)(hkUlong)klass.getSignature());
	m_classExternList += buffer;
	bool defineClass = false;
	for( int i = 0; i < m_originalClassList.getSize(); ++i )
	{
		if( &klass == m_originalClassList[i] )
		{
			defineClass = true;
			break;
		}
	}

	if( !defineClass )
	{
		return;
	}

	if( klass.getParent() )
	{
		HK_ASSERT3(0x215d081e, !klass.getParent()->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE),
			"Class '" << klass.getName() << "' is serializable, but parent class '" << klass.getParent()->getName() << "' is not serializable.");
		defineClassClass(*klass.getParent());
	}

	hkStringBuf parent = "HK_NULL";
	hkStringBuf interfaces = "HK_NULL";
	hkStringBuf enums = "HK_NULL";
	hkStringBuf members = "HK_NULL";
	hkStringBuf numMembers = "0";
	hkStringBuf defaults ="HK_NULL";

	hkStringBuf enumItemsDefinitionList;
	hkStringBuf enumsDefinitionList;
	hkStringBuf enumExtern;
	hkStringBuf classDefinitionList;

	if( klass.getParent() )
	{
		parent.printf("&%sClass", klass.getParent()->getName());
	}
	if( klass.getNumDeclaredInterfaces() > 0 )
	{
		//interfaces.printf("%sInterfaces", klass.getName());
	}
	if( klass.getNumDeclaredEnums() > 0 )
	{
		enums.printf("reinterpret_cast<const hkClassEnum*>(%sEnums)", klass.getName());

		buffer.printf(s_enumsStart, klass.getName());
		enumsDefinitionList += buffer;

		for( int e = 0; e < klass.getNumDeclaredEnums(); ++e )
		{
			const hkClassEnum& classEnum = klass.getDeclaredEnum(e);
			const char* completeEnumName = m_enumNameFromPointer.getWithDefault(&classEnum, HK_NULL);
			if( m_enumDoneFlagFromName.hasKey(completeEnumName) )
			{
				continue;
			}

			m_enumDoneFlagFromName.insert(completeEnumName, true);
			generateEnum(completeEnumName, classEnum, enumItemsDefinitionList);

			buffer.printf(s_enumDefinition, classEnum.getName(), completeEnumName, classEnum.getNumItems(), classEnum.getFlags().get() );
			enumsDefinitionList += buffer;

			buffer.printf(s_enumExtern, completeEnumName, klass.getName(), e);
			enumExtern += buffer;
		}
		enumsDefinitionList += s_enumsEnd;
	}
	enumsDefinitionList += enumExtern;

	// generate class
	classDefinitionList += enumItemsDefinitionList;
	classDefinitionList += enumsDefinitionList;

	if( klass.getNumDeclaredMembers() > 0 )
	{
		hkArray<int> memIdxWithDefault;
		hkTypedUnion defaultValue;

		members.printf("reinterpret_cast<const hkClassMember*>(%sClass_Members)", klass.getName());
		numMembers.printf("int(sizeof(%sClass_Members)/sizeof(hkInternalClassMember))", klass.getName());

		buffer.printf(s_membersStart, klass.getName());
		classDefinitionList += buffer;

		for( int i = 0; i < klass.getNumDeclaredMembers(); ++i )
		{
			hkStringBuf memClass("HK_NULL");
			hkStringBuf memEnum("HK_NULL");
			const hkInternalClassMember& mem = reinterpret_cast<const hkInternalClassMember&>(klass.getDeclaredMember(i));
			if( klass.getDefault(klass.getMemberIndexByName(mem.m_name), defaultValue) == HK_SUCCESS )
			{
				memIdxWithDefault.pushBack(i);
			}
			if( mem.m_class )
			{
				if( !mem.m_class->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
				{
					memClass.printf("&%sClass", mem.m_class->getName());
				}
				defineClassClass(*mem.m_class);
			}
			if( mem.m_enum )
			{
				char* enumName = m_enumNameFromPointer.getWithDefault(mem.m_enum, HK_NULL);
				HK_ASSERT(0x215d081b, enumName != HK_NULL );
				memEnum.printf("%sEnum", enumName);
			}
			const char* memTypeName;
			HK_ON_DEBUG(hkResult res = )m_classMemberEnumType->getNameOfValue(mem.m_type, &memTypeName);
			HK_ASSERT(0x215d081c, res == HK_SUCCESS);
			const char* memSubTypeName;
			HK_ON_DEBUG(res = )m_classMemberEnumType->getNameOfValue(mem.m_subtype, &memSubTypeName);
			HK_ASSERT(0x215d081d, res == HK_SUCCESS);
			buffer.printf(s_memberDefinition,
				mem.m_name, /* memberName */
				memClass.cString(), /* class pointer */
				memEnum.cString(), /* enum pointer */
				memTypeName/*hkClassMemberTypeEnumItems[mem.m_type].m_name*/, /* havokType */
				memSubTypeName/*hkClassMemberTypeEnumItems[mem.m_subtype].m_name*/, /* havokSubType */
				mem.m_cArraySize, /* cArraySize */
				mem.m_flags /* flags */
				);
			classDefinitionList += buffer;
		}
		classDefinitionList += s_membersEnd;
		if( memIdxWithDefault.getSize() > 0 )
		{
			defaults.printf("&%s_Default", klass.getName());
			generateDefaults(klass, memIdxWithDefault, buffer);
			classDefinitionList += buffer;
		}
	}
	buffer.printf(s_classDefinition,
		klass.getName(), /* hkClass variable name */
		klass.getName(), /* className */
		parent.cString(), /* parentClass */
		interfaces.cString(), /* implementedInterfaces */
		klass.getNumDeclaredInterfaces(), /* numImplementedInterfaces */
		enums.cString(), /* declaredEnums */
		klass.getNumDeclaredEnums(), /* numDeclaredEnums */
		members.cString(), /* declaredMembers */
		numMembers.cString(), /* numDeclaredMembers */
		defaults.cString()); /* defaults - variable name */
	classDefinitionList += buffer;
	m_classDefinitionList += classDefinitionList;
}


static void updateVariantInternal( void* variantPtr, int n, const hkClassNameRegistry& reg )
{
	hkVariant* v = static_cast<hkVariant*>(variantPtr);
	for( int i = 0; i < n; ++i )
	{
		if( v[i].m_class )
		{
			v[i].m_class = reg.getClassByName( v[i].m_class->getName() );
		}
	}
}

static void updateVariantInternalWithTracker( hkObjectUpdateTracker& tracker, void* variantPtr, int n, const hkClassNameRegistry& reg, hkPointerMap<const hkClass*, const hkClass*>& doneOldFromNewClass )
{
	hkVariant* v = static_cast<hkVariant*>(variantPtr);
	for( int i = 0; i < n; ++i )
	{
		if( v[i].m_class && !doneOldFromNewClass.hasKey(v[i].m_class) )
		{
			const hkClass* newClass = reg.getClassByName( v[i].m_class->getName() );
			if( newClass )
			{
				doneOldFromNewClass.insert(newClass, v[i].m_class);
				tracker.replaceObject(const_cast<hkClass*>(v[i].m_class), const_cast<hkClass*>(newClass), &hkClassClass);
				tracker.removeFinish(const_cast<hkClass*>(newClass));
			}
			else
			{
				tracker.replaceObject(const_cast<hkClass*>(v[i].m_class), HK_NULL, HK_NULL);
			}
			HK_ASSERT(0x2ed3f13f, v[i].m_class == newClass);
		}
	}
}

static void updateHomogeneousArrayInternalWithTracker( hkObjectUpdateTracker& tracker, hkClassMemberAccessor::HomogeneousArray& a, const hkClassNameRegistry& reg, hkPointerMap<const hkClass*, const hkClass*>& doneOldFromNewClass )
{
	if( a.klass && !doneOldFromNewClass.hasKey(a.klass) )
	{
		const hkClass* newClass = reg.getClassByName( a.klass->getName() );
		if( newClass )
		{
			doneOldFromNewClass.insert(newClass, a.klass);
			tracker.replaceObject(const_cast<hkClass*>(a.klass), const_cast<hkClass*>(newClass), &hkClassClass);
			tracker.removeFinish(const_cast<hkClass*>(newClass));
		}
		else
		{
			tracker.replaceObject(const_cast<hkClass*>(a.klass), HK_NULL, HK_NULL);
		}
	}
}

const char* HK_CALL hkVersionUtil::getDeprecatedCurrentVersion() 
{
	return "Havok-7.0.0-r1"; 
}


void HK_CALL hkVersionUtil::updateVariantClassPointers( void* pointer, const hkClass& klass, hkClassNameRegistry& reg, int numObjs )
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
					updateVariantInternal( maccess.asRaw(), nelem, reg );
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
						updateVariantInternal( array.data, array.size, reg );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
				}
				else if( member.getSubType() == hkClassMember::TYPE_STRUCT )
				{
					HK_ASSERT(0x2746110b, member.hasClass());
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
						updateVariantClassPointers( array.data, member.getStructClass(), reg, array.size );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
				}
				break;
			}
			case hkClassMember::TYPE_STRUCT:
			{
				HK_ASSERT(0x5a45c73f, member.hasClass());
				int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
				void* obj = pointer;
				int objCount = numObjs;
				while( --objCount >= 0 )
				{
					hkClassMemberAccessor maccess(obj, &member);
					updateVariantClassPointers( maccess.asRaw(), member.getStructClass(), reg, nelem );
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

static void updateVariantClassPointersWithTracker( hkObjectUpdateTracker& tracker, void* pointer, const hkClass& klass, hkClassNameRegistry& reg, int numObjs, hkPointerMap<const hkClass*, const hkClass*>& doneOldFromNewClass )
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
					updateVariantInternalWithTracker( tracker, maccess.asRaw(), nelem, reg, doneOldFromNewClass );
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
						updateVariantInternalWithTracker( tracker, array.data, array.size, reg, doneOldFromNewClass );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
				}
				else if( member.getSubType() == hkClassMember::TYPE_STRUCT )
				{
					HK_ASSERT(0x23aeddf8, member.hasClass());
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
						updateVariantClassPointersWithTracker( tracker, array.data, member.getStructClass(), reg, array.size, doneOldFromNewClass );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
				}
				break;
			}
			case hkClassMember::TYPE_STRUCT:
			{
				HK_ASSERT(0x339f5834, member.hasClass());
				int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
				void* obj = pointer;
				int objCount = numObjs;
				while( --objCount >= 0 )
				{
					hkClassMemberAccessor maccess(obj, &member);
					updateVariantClassPointersWithTracker( tracker, maccess.asRaw(), member.getStructClass(), reg, nelem, doneOldFromNewClass );
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

static void updateHomogeneousArrayClassPointersWithTracker( hkObjectUpdateTracker& tracker, void* pointer, const hkClass& klass, hkClassNameRegistry& reg, int numObjs, hkPointerMap<const hkClass*, const hkClass*>& doneOldFromNewClass )
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
			case hkClassMember::TYPE_HOMOGENEOUSARRAY:
			{
				void* obj = pointer;
				int objCount = numObjs;
				while( --objCount >= 0 )
				{
					hkClassMemberAccessor maccess(obj, &member);
					hkClassMemberAccessor::HomogeneousArray& array = maccess.asHomogeneousArray();
					updateHomogeneousArrayInternalWithTracker( tracker, array, reg, doneOldFromNewClass );
					obj = hkAddByteOffset(obj, klass.getObjectSize());
				}
				break;
			}
			case hkClassMember::TYPE_ARRAY:
			case hkClassMember::TYPE_SIMPLEARRAY:
			{
				if( member.getSubType() == hkClassMember::TYPE_STRUCT )
				{
					HK_ASSERT(0x1da7c311, member.hasClass());
					void* obj = pointer;
					int objCount = numObjs;
					while( --objCount >= 0 )
					{
						hkClassMemberAccessor maccess(obj, &member);
						hkClassMemberAccessor::SimpleArray& array = maccess.asSimpleArray();
						updateHomogeneousArrayClassPointersWithTracker( tracker, array.data, member.getStructClass(), reg, array.size, doneOldFromNewClass );
						obj = hkAddByteOffset(obj, klass.getObjectSize());
					}
				}
				break;
			}
			case hkClassMember::TYPE_STRUCT:
			{
				HK_ASSERT(0x73a8868f, member.hasClass());
				int nelem = member.getCstyleArraySize() ? member.getCstyleArraySize() : 1;
				void* obj = pointer;
				int objCount = numObjs;
				while( --objCount >= 0 )
				{
					hkClassMemberAccessor maccess(obj, &member);
					updateHomogeneousArrayClassPointersWithTracker( tracker, maccess.asRaw(), member.getStructClass(), reg, nelem, doneOldFromNewClass );
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

namespace
{
	struct ActionFromClassName : public hkStringMap<const hkVersionRegistry::ClassAction*>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE, ActionFromClassName);
		ActionFromClassName(
			const hkVersionRegistry::UpdateDescription& descHead,
			hkArray<hkVariant>& v,
			hkArray<int>::Temp& objectEditIndicesOut )
		{
			hkArray<const hkVersionRegistry::ClassAction*>::Temp actionChunks;
			const hkVersionRegistry::UpdateDescription* desc = &descHead;
			while( desc )
			{
				if( desc->m_actions )
				{
					actionChunks.insertAt(0, desc->m_actions);
				}
				desc = desc->m_next;
			}
			for( int i = 0; i < v.getSize(); ++i )
			{
				const hkClass* klass = v[i].m_class;
				const hkVersionRegistry::ClassAction* action = HK_NULL;
				if( this->get(klass->getName(), &action) != HK_SUCCESS )
				{
					action = findActionForClass(actionChunks, *klass);
					this->insert( klass->getName(), action );
				}
				if( action )
				{
					objectEditIndicesOut.pushBack(i);
				}
			}
		}

		const hkVersionRegistry::ClassAction* findActionForClass(const hkArrayBase<const hkVersionRegistry::ClassAction*>& actionChunks, const hkClass& classIn ) const
		{
			// we match all names in the hierarchy
			hkStringMap<const hkClass*> hierarchyNames;
			{
				const hkClass* c = &classIn;
				while(  c != HK_NULL )
				{
					hierarchyNames.insert(c->getName(), c);
					c = c->getParent();
				}
			}

			// the first match for any base class wins
			for( int i = 0; i < actionChunks.getSize(); ++i )
			{
				const hkVersionRegistry::ClassAction* action = actionChunks[i];
				for( ; action->oldClassName != HK_NULL; ++action )
				{
					if( /*const hkClass* c =*/ hierarchyNames.getWithDefault( action->oldClassName, HK_NULL ) )
					{
						//			HK_ON_DEBUG(hkUint32 loadedSig = c->getSignature());
						//			HK_ASSERT( 0x786cb087, (action->versionFlags & VERSION_REMOVED) || loadedSig == action->oldSignature );
						return action;
					}
				}
			}
			return HK_NULL;
		}
	};	
}


void* HK_CALL hkVersionUtil::copyObjects(const hkArrayBase<hkVariant>& objectsIn, hkArray<hkVariant>& objectsOut, hkObjectUpdateTracker& tracker )
{
	hkArray<int>::Temp objectIndices;
	objectIndices.setSize(objectsIn.getSize());
	for( int i = 0; i < objectsIn.getSize(); ++i )
	{
		objectIndices[i] = i;
	}
	return copySomeObjects(objectsIn, objectsOut, objectIndices, tracker);
}

//
// These update* functions would be more naturally placed in hkVersionRegistry, but
// we place them here so that if versioning is not used, the linker can strip all this
// code and the support functions.
//

hkResult HK_CALL hkVersionUtil::updateSingleVersion(
	hkArray<hkVariant>& objectsInOut,
	hkObjectUpdateTracker& tracker,
	const hkVersionRegistry::UpdateDescription& updateDescription,
	const hkClassNameRegistry* newClassRegistry)
{
	hkRenamedClassNameRegistry newClassFromOldName = hkRenamedClassNameRegistry( updateDescription.m_renames, newClassRegistry );
	const hkVersionRegistry::UpdateDescription* nextUpdateDescription = updateDescription.m_next;
	while( nextUpdateDescription )
	{
		newClassFromOldName.registerRenames(nextUpdateDescription->m_renames);
		nextUpdateDescription = nextUpdateDescription->m_next;
	}

	// indices of objects which need to be versioned
	hkArray<int>::Temp objectEditIndices; objectEditIndices.reserve(objectsInOut.getSize());
	ActionFromClassName actionFromClassName( updateDescription, objectsInOut, objectEditIndices );

	// additionally some objects need to be copied
	hkArray<int>::Temp objectCopyIndices; objectCopyIndices.reserve(objectsInOut.getSize());
	hkArray<int>::Temp removedIndices; removedIndices.reserve(objectsInOut.getSize());
	// save a copy of old objects
	hkArray<hkVariant>::Temp oldObjects;
	oldObjects = objectsInOut;
	{
		hkArray<int>::Temp withVariantIndices; withVariantIndices.reserve(objectEditIndices.getSize());
		hkArray<int>::Temp withHomogeneousArrayIndices; withHomogeneousArrayIndices.reserve(objectEditIndices.getSize());
		for( int editIndexIndex = 0; editIndexIndex < objectEditIndices.getSize(); ++editIndexIndex )
		{
			int editIndex = objectEditIndices[editIndexIndex];
			const char* oldClassName = objectsInOut[editIndex].m_class->getName();
			const hkVersionRegistry::ClassAction* action = actionFromClassName.getWithDefault(oldClassName, HK_NULL);
			HK_ASSERT(0x3b5dbb52, action != HK_NULL );
			const hkClass* newClass = newClassFromOldName.getClassByName( oldClassName );
			
			if( action->versionFlags & hkVersionRegistry::VERSION_REMOVED )
			{
				HK_ASSERT(0x41ea0ee9, newClass == HK_NULL );
				hkVariant& v = objectsInOut[editIndex];
				tracker.removeFinish( v.m_object );
				removedIndices.pushBack( editIndex );
			}
			else
			{
				objectsInOut[editIndex].m_class = newClass;
				if( !newClass )
				{
					HK_ASSERT3(0x29b61c04, false, "Class '" << oldClassName << "' needs to be copied but I can't find it.");
					return HK_FAILURE;
				}

				if( action->versionFlags & hkVersionRegistry::VERSION_MANUAL )
				{
					// manual
				}
				else if( action->versionFlags & hkVersionRegistry::VERSION_COPY )
				{
					objectCopyIndices.pushBack(editIndex);
				}
				else if( action->versionFlags & hkVersionRegistry::VERSION_INPLACE )
				{
					hkVariant& v = oldObjects[editIndex];
					hkVersionUtil::copyDefaults( v.m_object, *v.m_class, *newClass );
					tracker.removeFinish( v.m_object );
					tracker.addFinish( v.m_object, newClass->getName() );
				}

				if( action->versionFlags & hkVersionRegistry::VERSION_VARIANT )
				{
					withVariantIndices.pushBack( editIndex );
				}

				if( action->versionFlags & hkVersionRegistry::VERSION_HOMOGENEOUSARRAY )
				{
					withHomogeneousArrayIndices.pushBack( editIndex );
				}

				if( hkString::strCmp(newClass->getName(),oldClassName) != 0 )
				{
					hkVariant& v = oldObjects[editIndex];
					tracker.removeFinish( v.m_object );
					tracker.addFinish( v.m_object, newClass->getName() );
				}
			}
		}

		copySomeObjects( oldObjects, objectsInOut, objectCopyIndices, tracker );

		hkPointerMap<const hkClass*, const hkClass*> doneOldFromNewClass;
		for( int i = 0; i < withVariantIndices.getSize(); ++i )
		{
			int withVariantIndex = withVariantIndices[i];
			hkVariant& v = objectsInOut[withVariantIndex];
			updateVariantClassPointersWithTracker( tracker, v.m_object, *v.m_class, newClassFromOldName, 1, doneOldFromNewClass );
		}

		for( int i = 0; i < withHomogeneousArrayIndices.getSize(); ++i )
		{
			int withHomogeneousIndex = withHomogeneousArrayIndices[i];
			hkVariant& v = objectsInOut[withHomogeneousIndex];
			updateHomogeneousArrayClassPointersWithTracker( tracker, v.m_object, *v.m_class, newClassFromOldName, 1, doneOldFromNewClass );
		}
	}

	// Catch any classes which weren't edited above
	{
		int SENTINEL = objectsInOut.getSize()+1;
		int editIndexIndex = 0;
		int nextEdited = objectEditIndices.getSize() ? objectEditIndices[editIndexIndex] : SENTINEL;
		for( int i = 0; i < objectsInOut.getSize(); ++i )
		{
			if( i < nextEdited )
			{
				hkVariant& v = objectsInOut[i];
				if( const char* newName = newClassFromOldName.getRename( v.m_class->getName() ) )
				{
					tracker.removeFinish( v.m_object );
					tracker.addFinish( v.m_object, newName );
				}
				const char* oldClassName = v.m_class->getName();
				v.m_class = newClassFromOldName.getClassByName( oldClassName );
				if( !v.m_class )
				{
					HK_ASSERT3(0x138f548f, false, "Can't update to version '" << newClassRegistry->getName() << "', class '" << oldClassName << "' is not registered.");
					return HK_FAILURE;
				}
			}
			else if ( i == nextEdited )
			{
				editIndexIndex += 1;
				nextEdited = editIndexIndex < objectEditIndices.getSize()
					? objectEditIndices[editIndexIndex]
					: SENTINEL;
			}
			else
			{
				HK_ASSERT(0x5f1c66a4,0);
			}
		}
	}

	// Call the version functions
	{
		for( int editIndexIndex = 0; editIndexIndex < objectEditIndices.getSize(); ++editIndexIndex )
		{
			int editIndex = objectEditIndices[editIndexIndex];
			hkVariant& oldObj = oldObjects[editIndex];
			hkVariant& newObj = objectsInOut[editIndex];
			const hkVersionRegistry::ClassAction* action = actionFromClassName.getWithDefault( oldObj.m_class->getName(), HK_NULL );
			HK_ASSERT(0x138c549f, action != HK_NULL );
			if( action != HK_NULL && action->versionFunc != HK_NULL )
			{
				(action->versionFunc)( oldObj, newObj, tracker );
			}
		}
	}

	// update all objects to point to new versions
	{
		for( int copyIndexIndex = 0; copyIndexIndex < objectCopyIndices.getSize(); ++copyIndexIndex )
		{
			int copyIndex = objectCopyIndices[copyIndexIndex];
			tracker.replaceObject(
				oldObjects[copyIndex].m_object,
				objectsInOut[copyIndex].m_object,
				objectsInOut[copyIndex].m_class );
		}
		for( int removeIndexIndex = removedIndices.getSize()-1; removeIndexIndex >= 0; --removeIndexIndex )
		{
			int removeIndex = removedIndices[removeIndexIndex];
			HK_WARN(0x651f7aa5, "removing deprecated object of type " << oldObjects[removeIndex].m_class->getName());
			tracker.replaceObject( oldObjects[removeIndex].m_object, HK_NULL, HK_NULL );
			objectsInOut.removeAt( removeIndex );
		}
	}

	return HK_SUCCESS;
}

hkResult HK_CALL hkVersionUtil::updateBetweenVersions(
	hkArray<hkVariant>& objectsInOut,
	hkObjectUpdateTracker& tracker,
	const hkVersionRegistry& reg,
	const char* fromVersion,
	const char* toVersion )
{
	toVersion = toVersion ? toVersion : getDeprecatedCurrentVersion();

	hkArray<const hkVersionRegistry::Updater*>::Temp updatePath;
	if( reg.getVersionPath(fromVersion, toVersion, updatePath ) == HK_SUCCESS )
	{
		hkResult res = HK_SUCCESS;
		HK_SERIALIZE_LOG(("\nTrace(func=\"hkVersionUtil::updateBetweenVersions()\", fromVersion=\"%s\", toVersion=\"%s\")\n", fromVersion, toVersion));
		for( int pathIndex = 0; res == HK_SUCCESS && pathIndex < updatePath.getSize(); ++pathIndex )
		{
			if( updatePath[pathIndex]->optionalCustomFunction )
			{
				res = updatePath[pathIndex]->optionalCustomFunction( objectsInOut, tracker );
			}
			else
			{
				res = updateSingleVersion(objectsInOut, tracker, *updatePath[pathIndex]->desc, reg.getClassNameRegistry(updatePath[pathIndex]->toVersion));
			}
		}
		return res;
	}
	else
	{
		HK_WARN( 0x394c3ad7, "No version path from " << fromVersion << " to " << getDeprecatedCurrentVersion() );
		if( hkString::strCmp(fromVersion, getDeprecatedCurrentVersion()) >= 0 )
		{
			HK_WARN( 0x394c3ad8, 
				"You are trying to use the old versioning system but that only can version up to " << getDeprecatedCurrentVersion() << "\n" <<
				"Use hkSerializeUtil instead of calling versioning directly. It will handle the transition between new and old systems." );
		}
		return HK_FAILURE;
	}
}

hkResult HK_CALL hkVersionUtil::updateToVersion(
	hkPackfileReader& reader,
	const hkVersionRegistry& reg,
	const char* targetVersion )
{
	const char* originalVersion = reader.getContentsVersion();
	if( hkString::strCmp( originalVersion, targetVersion ) != 0 )
	{
		hkArray<hkVariant>& loadedObjects = reader.getLoadedObjects();
		if( loadedObjects.getSize() )
		{
			if( hkVersionUtil::updateBetweenVersions(
				loadedObjects,
				reader.getUpdateTracker(),
				reg,
				originalVersion,
				targetVersion ) == HK_SUCCESS )
			{
				reader.setContentsVersion( targetVersion );
				return HK_SUCCESS;
			}
		}
		return HK_FAILURE; // getContents or update failed
	}
	return HK_SUCCESS; // already latest version
}

hkResult HK_CALL hkVersionUtil::updateToCurrentVersion(
	hkPackfileReader& reader,
	const hkVersionRegistry& reg )
{
	return updateToVersion( reader, reg, getDeprecatedCurrentVersion() );
}

hkResult HK_CALL hkVersionUtil::generateCppExternClassList(hkOstream& os, const char* headerMacro, const hkClass*const* classesToGenerate, const char* registryVariableName)
{
	if( !classesToGenerate || !registryVariableName )
	{
		return HK_FAILURE;
	}

	if( headerMacro )
	{
		os.printf(s_classesHeaderBegin, headerMacro, headerMacro);
		os << _NEW_LINE_;
	}
	os << s_commonHeaders;
	int i;
	hkStringBuf version(hkVersionUtil::getDeprecatedCurrentVersion());
	version.replace("Havok","");
	version.replace("-","");
	version.replace(".","");

	// open the namespace
	os.printf(s_namespaceOpen, version.cString());
	// add common forward declarations
	os.printf(s_commonForwardDeclaration, registryVariableName);
	os << _NEW_LINE_;

	// write classes
	const hkClass* const* classes = classesToGenerate;
	// process classes one by one
	hkStringMap<hkBool32> classDoneFlagFromName;
	for( i = 0; classes && classes[i] ; ++i )
	{
		const hkClass& klass = *classes[i];
		if( classDoneFlagFromName.hasKey(klass.getName()) )
		{
			continue;
		}
		else
		{
			classDoneFlagFromName.insert(klass.getName(), true);
			if( !klass.getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
			{
				os.printf(s_externClass, classes[i]->getName(), (void*)(hkUlong)classes[i]->getSignature());
			}
		}
	}
	// close the namespace
	os.printf(s_namespaceClose, version.cString());

	if( headerMacro )
	{
		os.printf(s_classesHeaderEnd, headerMacro);
		os << _NEW_LINE_;
	}
	return HK_SUCCESS;
}

hkResult HK_CALL hkVersionUtil::generateCppClassList(hkOstream& os, const hkClass*const* classesToGenerate, const char* pchfilename, const char* registryVariableName)
{
	if( !classesToGenerate || !registryVariableName )
	{
		return HK_FAILURE;
	}

	// write the headers
	if( pchfilename )
	{
//		os << "#include <";
		os << pchfilename;
		os << ">"   _NEW_LINE_;
	}
	os << s_commonHeaders;
	hkStringBuf buffer;
	hkStringBuf version(hkVersionUtil::getDeprecatedCurrentVersion());
	version.replace("Havok","");
	version.replace("-","");
	version.replace(".","");

	// open the namespace
	os.printf(s_namespaceOpen, version.cString());
	// add common forward declarations
	os.printf(s_commonForwardDeclaration, registryVariableName);
	os << _NEW_LINE_;

	// write classes
	hkInt32 numOfClasses = getNumElements<const hkClass>(const_cast<const hkClass**>(classesToGenerate));
	hkArray<const hkClass*> originalClassList(const_cast<const hkClass**>(classesToGenerate), numOfClasses, numOfClasses);
	// collect all used enums in classes
	hkPointerMap<const hkClassEnum*, char*> enumNameFromPointer;
	hkStringMap<hkBool32> enumDoneFlagFromName;
	hkStringBuf enumForward;
	{
		hkStringBuf completeEnumName;
		for( int i = 0; i < originalClassList.getSize() ; ++i )
		{
			const hkClass& klass = *originalClassList[i];
			for( int e = 0; e < klass.getNumDeclaredEnums(); ++e )
			{
				const hkClassEnum& classEnum = klass.getDeclaredEnum(e);
				if( enumNameFromPointer.hasKey(&classEnum) )
				{
					continue;
				}
				completeEnumName.printf("%s%s", klass.getName(), classEnum.getName());
				enumNameFromPointer.insert(&classEnum, hkString::strDup(completeEnumName.cString()));
				buffer.printf(s_enumForward, completeEnumName.cString());
				enumForward += buffer;
			}
		}
	}
	// collect all used enums in class members
	hkBool32 foundGlobalEnum = false;
	hkStringBuf enumItemsDefinitionList;
	hkStringBuf enumsDefinitionList;
	hkStringBuf enumExtern;
	hkStringBuf classExternList;
	hkStringBuf classDefinitionList;
	hkInt32 globalEnumIdx = 0;
	for( int i = 0; i < originalClassList.getSize() ; ++i )
	{
		const hkClass& klass = *originalClassList[i];
		if( klass.getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
		{
			continue;
		}
		for( int m = 0; m < klass.getNumDeclaredMembers(); ++m )
		{
			const hkInternalClassMember& mem = reinterpret_cast<const hkInternalClassMember&>(klass.getDeclaredMember(m));
			if( mem.m_enum )
			{
				if( enumNameFromPointer.hasKey(mem.m_enum) )
				{
					continue;
				}
				// global enum
				const hkClassEnum& classEnum = *mem.m_enum;
				char* completeEnumName = hkString::strDup(classEnum.getName());
				enumNameFromPointer.insert(&classEnum, completeEnumName);
				enumDoneFlagFromName.insert(completeEnumName, true);
				if( !foundGlobalEnum )
				{
					foundGlobalEnum = true;
					buffer.printf(s_enumsStart, "");
					enumsDefinitionList += buffer;
				}
				generateEnum(completeEnumName, classEnum, enumItemsDefinitionList);
				buffer.printf(s_enumDefinition, classEnum.getName(), completeEnumName, classEnum.getNumItems(), classEnum.getFlags().get() );
				enumsDefinitionList += buffer;

				buffer.printf(s_enumExtern, completeEnumName, "", globalEnumIdx++);
				enumExtern += buffer;
			}
		}
	}
	if( foundGlobalEnum )
	{
		enumsDefinitionList += s_enumsEnd;
		enumsDefinitionList += enumExtern;
		classDefinitionList += enumItemsDefinitionList;
		classDefinitionList += enumsDefinitionList;
	}
	// process classes one by one
	CollectClassDefinitions classDef(originalClassList, enumNameFromPointer, enumDoneFlagFromName);
	for(int i = 0; i < originalClassList.getSize(); ++i )
	{
		classDef.defineClassClass(*originalClassList[i]);
	}

	os << classDef.getClassExternList();
	os << _NEW_LINE_;
	if( enumForward.getLength() > 0 )
	{
		os << enumForward.cString();
		os << _NEW_LINE_;
	}
	os << classDef.getClassDefinitionList();

	classDefinitionList = "";
	for( int i = 0; i < originalClassList.getSize(); ++i )
	{
		if( originalClassList[i]->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
		{
			continue;
		}
		buffer.printf("\t\t&%sClass,"   _NEW_LINE_, originalClassList[i]->getName());
		classDefinitionList += buffer;
	}
	os.printf(s_classStaticRegistry, classDefinitionList.cString(), registryVariableName);

	// close the namespace
	os.printf(s_namespaceClose, version.cString());

	// free memory
	hkPointerMap<const hkClassEnum*, char*>::Iterator iter = enumNameFromPointer.getIterator();
	while( enumNameFromPointer.isValid(iter) )
	{
		char* enumName = enumNameFromPointer.getValue(iter);
		hkDeallocate<char>(enumName);
		iter = enumNameFromPointer.getNext(iter);
	}
	return HK_SUCCESS;
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
