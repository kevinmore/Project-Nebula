/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Util/hkVersionCheckingUtils.h>
#include <Common/Base/Config/hkConfigBranches.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/System/Io/Writer/hkStreamWriter.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/Config/hkConfigVersion.h>

#if 0
//#include <Common/Base/Fwd/hkcstdio.h>
#	define TRACE(A) printf A
#else
#	define TRACE(A)
#endif

extern const hkClassEnum* hkClassMemberTypeEnum;

namespace
{
	typedef hkStringMap<const hkClass*> Map;

	void makeMap( Map& m, hkArray<const hkClass*>& l)
	{
		for( int i = l.getSize() - 1; i >= 0; --i )
		{
			if(l[i])
			{
				m.insert( l[i]->getName(), l[i] );
			}
			else
			{
				l.removeAt(i);
			}
		}
	}

	void makeClassArray(const hkClass* const * nullTerminatedClassArray, hkArray<const hkClass*>& classArray)
	{
		const hkClass* const * current = nullTerminatedClassArray;
		while( *current )
		{
			classArray.pushBack(*current);
			++current;
		}
	}

	class EmptyStreamWriter : public hkStreamWriter
	{
		virtual hkBool isOk() const { return true; }
		virtual int write(const void* buf, int nbytes) { return nbytes; }
	};
}

void HK_CALL hkVersionCheckingUtils::summarizeChanges(hkOstream& output, const hkClass& c0, const hkClass& c1, bool detailed )
{

	if ( (detailed) && ( c0.getObjectSize() != c1.getObjectSize() ) )
	{
		output.printf("\tobject sizes differ %i %i\n", c0.getObjectSize(), c1.getObjectSize() );
	}

	// Check for removed enumerations
	{
		int c0Enums = c0.getNumEnums();
		for (int e0=0; e0 < c0Enums; ++e0 )
		{
			const hkClassEnum& c0e = c0.getEnum( e0 );
			const hkClassEnum* c1ep = c1.getEnumByName( c0e.getName() );
			if (c1ep == HK_NULL)
			{
				output.printf( "\tenum '%s' removed\n", c0e.getName() );
			}
		}
	}

	// Check for added enumerations 
	{
		int c1Enums = c1.getNumEnums();
		for (int e1=0; e1 < c1Enums; ++e1 )
		{
			const hkClassEnum& c1e = c1.getEnum( e1 );
			const hkClassEnum* c0ep = c0.getEnumByName( c1e.getName() );
			if (c0ep == HK_NULL)
			{
				output.printf( "\tenum '%s' added\n", c1e.getName() );
			}
		}
	}

	// Checked for changed enumerations
	{
		int c1Enums = c1.getNumEnums();
		for (int e1=0; e1 < c1Enums; ++e1 )
		{
			const hkClassEnum& c1e = c1.getEnum( e1 );
			const hkClassEnum* c0ep = c0.getEnumByName( c1e.getName() );
			if ((c0ep != HK_NULL) && (c1e.getSignature() != c0ep->getSignature()) )
			{
				// Enumeration changed
				output.printf( "\tenum '%s' changed\n", c1e.getName() );

				// Added and changed
				{
					int c1Vals = c1e.getNumItems();
					for (int v1=0; v1 < c1Vals; ++v1)
					{
						int c0eVal;
						if ( c0ep->getValueOfName(c1e.getItem( v1 ).getName(), &c0eVal) == HK_FAILURE )
						{
							output.printf( "\t\tvalue added '%s'\n", c1e.getItem( v1 ).getName() );
						}
						else if ( c1e.getItem( v1 ).getValue() != c0eVal )
						{
							output.printf( "\t\tvalue changed '%s'\n",  c1e.getItem( v1 ).getName() );
						}
					}
				}

				// Removed
				{
					int c0Vals = c0ep->getNumItems();
					for (int v0=0; v0 < c0Vals; ++v0)
					{
						int c1eVal;
						if ( c1e.getValueOfName(c0ep->getItem( v0 ).getName(), &c1eVal) == HK_FAILURE )
						{
							output.printf( "\tEnumeration '%s' : value removed '%s'\n", c1e.getName(), c0ep->getItem( v0 ).getName() );
						}
					}
				}
			}
		}
	}

	// Check changed members
	int c1numMembers = c1.getNumMembers();
	for( int i1 = 0; i1 < c1numMembers; ++i1 )
	{
		const hkClassMember& c1m = c1.getMember( i1 );
		if( const hkClassMember* c0mp = c0.getMemberByName( c1m.getName() ) )
		{
			if ( detailed && ( &c1m.getStructClass() || &c0mp->getStructClass() ) )
			{
				const hkClass& c1s = c1m.getStructClass();
				const hkClass& c0s = c0mp->getStructClass();
				if( (&c1s) && (&c0s) )
				{
					if( c1s.getSignature() != c0s.getSignature() )
					{
						output.printf("\tdefinition of '%s' %s changed\n", c1m.getName(), c1s.getName() );
					}						
				}
				else
				{
					const hkClass& c = &c1s ? c1s : c0s;
					output.printf("\t?? new definition of '%s' %s\n", c1m.getName(), c.getName() );
				}
			}
			if ( detailed && ( c1m.getOffset() != c0mp->getOffset() ) )
			{
				output.printf("\toffset of '%s' changed %i %i\n", c1m.getName(), c0mp->getOffset(), c1m.getOffset() );
			}
			if( c1m.getSizeInBytes() != c0mp->getSizeInBytes() )
			{
				output.printf("\tsize of '%s' changed %i %i\n", c1m.getName(), c0mp->getSizeInBytes(), c1m.getSizeInBytes() );
			}
			if( c1m.getType() != c0mp->getType() || c1m.getSubType() != c0mp->getSubType())
			{
				const char* c0t = HK_NULL;
				const char* c0s = HK_NULL;
				const char* c1t = HK_NULL;
				const char* c1s = HK_NULL;
				hkClassMemberTypeEnum->getNameOfValue( c0mp->getType(), &c0t );
				hkClassMemberTypeEnum->getNameOfValue( c0mp->getSubType(), &c0s );
				hkClassMemberTypeEnum->getNameOfValue( c1m.getType(), &c1t );
				hkClassMemberTypeEnum->getNameOfValue( c1m.getSubType(), &c1s );

				output.printf("\ttype of '%s' changed %s.%s %s.%s\n", c1m.getName(), c0t,c0s, c1t,c1s);
			}
			if( c1m.getCstyleArraySize() != c0mp->getCstyleArraySize() )
			{
				output.printf("\tarray size of '%s' changed %i %i\n", c1m.getName(), c1m.getCstyleArraySize(), c0mp->getCstyleArraySize());
			}
			if( c1m.getFlags() != c0mp->getFlags() )
			{
				output.printf("\tflags of '%s' changed %i %i\n", c1m.getName(), c1m.getFlags().get(), c0mp->getFlags().get());
			}
		}
		else
		{
			

			EmptyStreamWriter nullWriter;
			bool hasDefault  = c1.getDefault( i1, &nullWriter ) == HK_SUCCESS;

			output.printf("\tmember '%s' added : has default %s\n", c1m.getName(), hasDefault ? "yes" : "no");
		}
	}
	int c0numMembers = c0.getNumMembers();
	for( int i0 = 0; i0 < c0numMembers; ++i0 )
	{
		const hkClassMember& c0m = c0.getMember( i0 );
		if( c1.getMemberByName( c0m.getName() ) == HK_NULL )
		{
			output.printf("\tmember '%s' removed\n", c0m.getName() );
		}
	}

}



namespace
{
	enum Dependency
	{
		HK_DEPENDENCY_HAS_NO_CHANGES,
		HK_DEPENDENCY_CHANGED,
		HK_DEPENDENCY_NOT_SURE
	};
	struct DependencyDetail
	{
		hkEnum<Dependency, hkInt32> flag;
		hkInt32 version;

		operator hkUint64() { return (((hkUint64)flag) << 32) | (hkUint64)version; }
		void operator =(hkUint64 v)
		{
			flag = (Dependency)(v >> 32);
			version = (hkInt32)(v & 0xFFFFFFFF);
		}
	};
	typedef hkStringMap<int> DependenciesMap;

	static const char* getVersionString(int version, hkStringBuf& buf)
	{
		buf.clear();
		int verHigh = version & 0xFFFF0000;
		int verLow = version & 0x0000FFFF;
		if( verHigh != 0 )
		{
			switch( verHigh )
			{
				case HK_BRANCH_PRE_60:
				{
					buf = "HK_BRANCH_PRE_60 | ";
					break;
				}
				default:
				{
					buf.printf("0x%p | ", (void*)(hkUlong)verHigh);
				}
			}
		}
		buf.appendPrintf("%d", (void*)(hkUlong)verLow);
		return buf.cString();
	}
#define HK_VER(V,buf) getVersionString(V,buf)

	const char sPatchParentSet[] =			"HK_PATCH_PARENT_SET";
	const char sPatchGroup[] =				"HK_PATCH_GROUP";
	const char sPatchDepends[] =			"HK_PATCH_DEPENDS";
	const char sPatchEnumAdded[] =			"HK_PATCH_ENUM_ADDED";
	const char sPatchEnumRemoved[] =		"HK_PATCH_ENUM_REMOVED";
	const char sPatchEnumRenamed[] =		"HK_PATCH_ENUM_RENAMED";
	const char sPatchEnumValueAdded[] =		"HK_PATCH_ENUM_VALUE_ADDED";
	const char sPatchEnumValueRemoved[] =	"HK_PATCH_ENUM_VALUE_REMOVED";
	const char sPatchEnumValueChanged[] =	"HK_PATCH_ENUM_VALUE_CHANGED";
	const char sPatchMemberRemoved[] =		"HK_PATCH_MEMBER_REMOVED";
	const char sPatchMemberAdded[] =		"HK_PATCH_MEMBER_ADDED";
	const char sPatchMemberAddedInt[] =		"HK_PATCH_MEMBER_ADDED_INT";
	const char sPatchMemberAddedReal[] =	"HK_PATCH_MEMBER_ADDED_REAL";
	const char sPatchMemberAddedByte[] =	"HK_PATCH_MEMBER_ADDED_BYTE";
	const char sPatchMemberAddedVec4[] =	"HK_PATCH_MEMBER_ADDED_VEC_4";
	const char sPatchMemberAddedVec12[] =	"HK_PATCH_MEMBER_ADDED_VEC_12";
	const char sPatchMemberAddedVec16[] =	"HK_PATCH_MEMBER_ADDED_VEC_16";
	const char sPatchMemberAddedCString[] = "HK_PATCH_MEMBER_ADDED_CSTRING";
	const char sPatchMemberAddedPointer[] = "HK_PATCH_MEMBER_ADDED_POINTER";
	const char sPatchMemberDefaultSetInt[] =	"HK_PATCH_MEMBER_DEFAULT_SET_INT";
	const char sPatchMemberDefaultSetReal[] =	"HK_PATCH_MEMBER_DEFAULT_SET_REAL";
	const char sPatchMemberDefaultSetByte[] =	"HK_PATCH_MEMBER_DEFAULT_SET_BYTE";
	const char sPatchMemberDefaultSetVec4[] =	"HK_PATCH_MEMBER_DEFAULT_SET_VEC_4";
	const char sPatchMemberDefaultSetVec12[] =	"HK_PATCH_MEMBER_DEFAULT_SET_VEC_12";
	const char sPatchMemberDefaultSetVec16[] =	"HK_PATCH_MEMBER_DEFAULT_SET_VEC_16";
	const char sPatchMemberDefaultSetCString[] =	"HK_PATCH_MEMBER_DEFAULT_SET_CSTRING";
	const char sPatchMemberDefaultSetPointer[] =	"HK_PATCH_MEMBER_DEFAULT_SET_POINTER";
	const char sPatchMemberRenamed[] =		"HK_PATCH_MEMBER_RENAMED";
	const char sPatchMemberDefault[] =		"HK_PATCH_MEMBER_DEFAULT";

#define STREAM_PATCH_VERSION(VERSION) \
	VERSION

	static inline int getVersion(const hkClass* c)
	{
		return c ? c->getDescribedVersion() : 0;
	}

	static inline int getVersion(const hkDataClass& c)
	{
		return c.isNull() ? 0 : c.getVersion();
	}

	static const char* getVecType(hkDataObject::Type type)
	{
		if (type->isTuple())
		{
			hkDataObject::Type parent = type->getParent();
			if (parent->isReal())
			{
				switch (type->getTupleSize())
				{
					case 4:		return "VEC_4";	
					case 8:		return "VEC_8";
					case 12:	return "VEC_12";
					case 16:	return "VEC_16";
					break;
				}
			}
		}
		return HK_NULL;
	}

	static const char* getStringFromType(hkDataObject::Type type, hkStringBuf& buf, int& tupleOut)
	{
		tupleOut = 0;

		buf.clear();
		buf.append("TYPE");

		if (type->isArray())
		{
			buf.append("_ARRAY");
			type = type->getParent();
		}

		if (type->isTuple())
		{
			// Handle Vec types and tuples of vec type
			const char* vecType = getVecType(type);
			if (vecType)
			{
				buf.append("_");
				buf.append(vecType);
				return buf;
			}
			hkDataObject::Type parent = type->getParent();
			vecType = getVecType(parent);
			if (vecType)
			{
				tupleOut = type->getTupleSize();
				buf.append("_TUPLE_");
				buf.append(vecType);
				return buf;
			}

			tupleOut = type->getTupleSize();

			buf.append("_TUPLE");
			type = parent;
		}
		
		switch (type->getSubType())
		{
			case hkTypeManager::SUB_TYPE_BYTE:
			{
				buf.append("_BYTE");
				break;
			}
			case hkTypeManager::SUB_TYPE_INT:
			{
				buf.append("_INT");
				break;
			}
			case hkTypeManager::SUB_TYPE_REAL:
			{
				buf.append("_REAL");
				break;
			}
			case hkTypeManager::SUB_TYPE_CSTRING:
			{
				buf.append("_CSTRING");
				break;
			}
			case hkTypeManager::SUB_TYPE_CLASS:
			{
				buf.append("_STRUCT");
				break;
			}
			case hkTypeManager::SUB_TYPE_POINTER:
			{
				if (type->getParent()->isClass())
				{
					buf.append("_OBJECT");
				}
				else
				{
					HK_ASSERT(0x3242343, !"Unable to convert type");
					return HK_NULL;
				}
				break;
			}
			default:
			{
				HK_ASSERT(0x2432a423, !"Unable to convert type");
				return HK_NULL;
			}
		}

		// Return the buffer
		return buf;
	}

	static inline void reportEnumValueAdded(hkOstream& output, const hkClassEnum& e, int idx)
	{
		output << "\t" << sPatchEnumValueAdded << "(\"" << e.getName() << "\", \"" << e.getItem(idx).getName() << "\", " << e.getItem(idx).getValue() << ")\n";
	}

	static inline void reportEnumValueChanged(hkOstream& output, const hkClassEnum& e0, int i0, const hkClassEnum& e1, int i1)
	{
		output << "\t" << sPatchEnumValueChanged << "(\"" << e0.getName() << "\", \"" << e0.getItem(i0).getName() << "\", " << e0.getItem(i0).getValue() << ", " << e1.getItem(i1).getValue() << ")\n";
	}

	static inline void reportEnumValueRemoved(hkOstream& output, const hkClassEnum& e, int idx)
	{
		output << "\t" << sPatchEnumValueRemoved << "(\"" << e.getName() << "\", \"" << e.getItem(idx).getName() << "\", " << e.getItem(idx).getValue() << ")\n";
	}

	static inline void reportEnumRemoved(hkOstream& output, const hkClass& c, const hkClassEnum& e)
	{
		for( int i = 0; i < e.getNumItems(); ++i )
		{
			reportEnumValueRemoved(output, e, i);
		}
		output << "\t" << sPatchEnumRemoved << "(\"" << e.getName() << "\")\n";
	}

	static inline void reportEnumAdded(hkOstream& output, const hkClass& c, const hkClassEnum& e)
	{
		output << "\t" << sPatchEnumAdded << "(\"" << e.getName() << "\")\n";
	}

	static const char* REPORT_TYPE_NAME(const hkDataClass::MemberInfo& mem, hkStringBuf& buf)
	{
		hkDataObject::Type type = mem.m_type->findTerminal();

		if( type->isClass())
		{
			buf.printf("\"%s\"", type->getTypeName());
		}
		else
		{
			buf = "HK_NULL";
		}
		return buf.cString();
		
	}

	static const char* REPORT_TYPE_VAL(hkTypedUnion& val, const hkClassMember::Type type, hkStringBuf& buf)
	{
		switch(type)
		{
		case hkClassMember::TYPE_INT8:
			buf.printf("%d", hkInt32(val.getStorage().m_int8)); break;
		case hkClassMember::TYPE_UINT8:
		case hkClassMember::TYPE_ENUM:
		case hkClassMember::TYPE_BOOL:
			buf.printf("%d", hkUint32(val.getStorage().m_uint8)); break;
		case hkClassMember::TYPE_INT16:
		case hkClassMember::TYPE_HALF:
			buf.printf("%d", hkInt32(val.getStorage().m_int16)); break;
		case hkClassMember::TYPE_UINT16:
			buf.printf("%d", hkInt32(val.getStorage().m_uint16)); break;
		case hkClassMember::TYPE_INT32:
			buf.printf("%d", val.getStorage().m_int32); break;
		case hkClassMember::TYPE_UINT32:
#if HK_POINTER_SIZE == 4
		case hkClassMember::TYPE_ULONG:
#endif
			buf.printf("%d", val.getStorage().m_uint32); break;
		case hkClassMember::TYPE_INT64:
			buf.printf("%d", val.getStorage().m_int64); break;
		case hkClassMember::TYPE_UINT64:
#if HK_POINTER_SIZE == 8
		case hkClassMember::TYPE_ULONG:
#endif
			buf.printf("%d", val.getStorage().m_uint64); break;
		case hkClassMember::TYPE_REAL:
			buf.printf("%ff", val.getStorage().m_float); break;
		case hkClassMember::TYPE_VECTOR4:
		case hkClassMember::TYPE_QUATERNION:
			{
				const float* vals = val.getStorage().m_float4;
				buf.printf("%ff,%ff,%ff,%ff", vals[0], vals[1], vals[2], vals[3]); break;
			}
		case hkClassMember::TYPE_QSTRANSFORM:
		case hkClassMember::TYPE_MATRIX3:
			{
				const float* vals = val.getStorage().m_float12;
				buf.printf("%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff",
					vals[0], vals[1], vals[2], vals[3],
					vals[4], vals[5], vals[6], vals[7],
					vals[8], vals[9], vals[10], vals[11]); break;
			}
		case hkClassMember::TYPE_TRANSFORM:
		case hkClassMember::TYPE_MATRIX4:
			{
				const float* vals = val.getStorage().m_float16;
				buf.printf("%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff,%ff", 
					vals[0], vals[1], vals[2], vals[3],
					vals[4], vals[5], vals[6], vals[7],
					vals[8], vals[9], vals[10], vals[11],
					vals[12], vals[13], vals[14], vals[15]); break;
			}
		case hkClassMember::TYPE_CSTRING:
		case hkClassMember::TYPE_STRINGPTR:
			{
				const char* ptr = reinterpret_cast<const char*>(val.getStorage().m_ulong);
				if(ptr)
				{
					buf.printf("\"%s\"", ptr);
				}
				else
				{
					buf.printf("HK_NULL");
				}
			}
			break;
		case hkClassMember::TYPE_POINTER:
			buf.printf("%ld", reinterpret_cast<hkUlong>(reinterpret_cast<const char*>(val.getStorage().m_ulong))); break;
		default:
			HK_ASSERT2(0x40a9dcc3, 0, "Invalid patch default");
		}
		return buf.cString();
	}

	static const char* REPORT_CLASS_NAME(const hkDataClass& obj, hkStringBuf& buf)
	{
		if( !obj.isNull() )
		{
			buf.printf("\"%s\"", obj.getName());
		}
		else
		{
			buf = "HK_NULL";
		}
		return buf.cString();
	}

	static inline bool localHierarchy(const hkDataClass& klass, const hkDataClass& dependClass)
	{
		if( hkString::strCmp(klass.getName(), dependClass.getName()) == 0
			&& klass.getVersion() == dependClass.getVersion() )
		{
			return true;
		}
		else if( !klass.getParent().isNull() )
		{
			return localHierarchy(klass.getParent(), dependClass);
		}
		return false;
	}

	static inline void setHierarchyDependencies(const hkDataClass& c, const hkDataClass& dependClass, DependenciesMap& dependecies)
	{
		if( dependecies.hasKey(dependClass.getName()) )
		{
			return;
		}

		if( c.isNull() || /*!localHierarchy(*c, dependClass)*/hkString::strCmp(c.getName(), dependClass.getName()) != 0 )
		{
			DependencyDetail d = { HK_DEPENDENCY_NOT_SURE, dependClass.getVersion() };
			dependecies.insert(dependClass.getName(), d.version);
		}
		if( !dependClass.getParent().isNull() )
		{
			setHierarchyDependencies(c, dependClass.getParent(), dependecies);
		}
	}
/*
	static inline void reportMemberDefault(hkOstream& output, const hkClass* c0, int i0, const hkClass* c1, int i1)
	{
		hkString defVal0;
		getDefaultValueAsString(c0, i0, defVal0); // e.g. "{-1}" or "{0.00000,1.00000,1.00000,1.00000}"
		hkString defVal1;
		getDefaultValueAsString(c1, i1, defVal1);
		const hkClass* c = c0 ? c0 : c1;
		int i = c0 ? i0 : i1;
		HK_ASSERT(0x4fcfffba, c);
		output << "\t" << sPatchMemberDefault << "(\"" << c->getDeclaredMember(i).getName() << "\", " << defVal0.cString() << ", " << defVal1.cString() << ")\n";
	}
*/

#define HK_DATATYPE_OK(T) (!(T)->isVoid())
//#define HK_OBJECT_TYPE_OK(T,PREFIX) ((T) == hkDataObject::TYPE##PREFIX##OBJECT || (T) == hkDataObject::TYPE##PREFIX##STRUCT)
//#define HK_DATAOBJECT_OK(T) ((T)->isClass() || ((T)->isPointer() && (T)->getParent()->isClass()) || (T(HK_OBJECT_TYPE_OK(T,_) || HK_OBJECT_TYPE_OK(T,_ARRAY_) || HK_OBJECT_TYPE_OK(T,_TUPLE_))

	HK_FORCE_INLINE static hkBool HK_CALL _isObjectOrStruct(hkDataObject::Type type)
	{
		return type->isClass() ||
				(type->isPointer() && type->getParent()->isClass());
	}

	HK_FORCE_INLINE static hkBool HK_CALL _isDataOk(hkDataObject::Type type)
	{
		return _isObjectOrStruct(type) ||
			   (type->isArray() && _isObjectOrStruct(type->getParent())) ||
			   (type->isTuple() && _isObjectOrStruct(type->getParent()));
	}
#define HK_DATAOBJECT_OK(T)	_isDataOk(T)

	struct DependencyInfo 
	{
		const char* typeName;
		int version;

		bool operator ==(const DependencyInfo& v)
		{
			return (version == v.version)
				&& (typeName == v.typeName
				|| (typeName && v.typeName && hkString::strCmp(typeName, v.typeName) == 0 ) );
		}

		bool operator !=(const DependencyInfo& v)
		{
			return !(*this == v);
		}
	};

	class GenericClassCollector
	{
		public:
			GenericClassCollector(const hkDataWorld& worldIn, const hkClassNameRegistry& classReg, const hkVersionPatchManager& patchManager) : m_patchManager(patchManager)
			{
				// add classes from registry to m_world
				{
					m_world.setClassRegistry(&classReg);
				}

				// copy classes from 'worldIn' to m_manifestWorld
				{
					hkArray<hkDataClassImpl*>::Temp classes;
					worldIn.findAllClasses(classes);
					for( int i = 0; i < classes.getSize(); ++i )
					{
						m_manifestWorld.copyClassFromWorld(classes[i]->getName(), worldIn);
					}
				}
				// find all classes with the HK_CLASS_ADDED patch
				// and upgrade manifest classes to the most recent version using patches,
				// also collect source/destination patches
				const hkArray<const hkVersionPatchManager::PatchInfo*>& patches = m_patchManager.getPatches();
				for( int i = 0; i < patches.getSize(); ++i )
				{
					const hkVersionPatchManager::PatchInfo* patch = patches[i];
					// make map of source patches
					if( patch->oldVersion != -1/*HK_CLASS_ADDED*/ )
					{
						HK_ASSERT(0x206e6820, patch->oldName);
						hkUint64 uid = m_patchManager.getUid(patch->oldName, patch->oldVersion);
						HK_ASSERT3(0x206f6620, m_srcPatchFromUid.hasKey(uid) == false, "Patch error. Found duplicated patch for class " << patch->oldName << " (version " << (void*)(hkUlong)patch->oldVersion << ")." );
						m_srcPatchFromUid.insert(uid, patch);
					}
					// make map of destination patches
					if( patch->newVersion != -1/*HK_CLASS_REMOVED*/ )
					{
						const char* className = patch->newName ? patch->newName : patch->oldName;
						hkUint64 uid = m_patchManager.getUid(className, patch->newVersion);
						HK_ASSERT(0x49ccfdc3, m_dstPatchFromUid.hasKey(uid) == false);
						m_dstPatchFromUid.insert(uid, patch);
					}
				}
				// upgrade manifest classes to the most recent version using all patches
				patchManager.applyPatchesDebug(m_manifestWorld);
				{
					hkArray<hkDataClassImpl*>::Temp classes;
					m_manifestWorld.findAllClasses(classes);
					for( int i = 0; i < classes.getSize(); ++i )
					{
						hkDataClass c(classes[i]);
						m_manifestClassFromName.insert(c.getName(), classes[i]);
					}
				}
			}

			hkDataClass getAndMarkManifestClass(const char* name) const
			{
				hkDataClassImpl* manifestClass = m_manifestWorld.findClass(name);
				if( manifestClass )
				{
					m_manifestClassFromName.remove(manifestClass->getName());
					return manifestClass;
				}

				// the class was renamed, but no patch found
				return HK_NULL;
			}

			void getUnmarkedManifestClasses(hkArray<hkDataClassImpl*>& classesOut) const
			{
				for( hkStringMap<hkDataClassImpl*>::Iterator it = m_manifestClassFromName.getIterator();
						m_manifestClassFromName.isValid(it); it = m_manifestClassFromName.getNext(it) )
				{
					classesOut.pushBack(m_manifestClassFromName.getValue(it));
				}
			}

			inline int findNewOrReuseVersion(const char* name, int version) const
			{
				hkUint64 uid = m_patchManager.getUid(name, version);
				int pindex = m_patchManager.findLastPatchIndexForUid(uid); // returns -1 if class still exists
				const hkVersionPatchManager::PatchInfo* p = pindex != -1 ? m_patchManager.getPatch(pindex) : m_dstPatchFromUid.getWithDefault(uid, HK_NULL);
				if( p )
				{
					int versionFromPatch = p->newVersion != -1 ? p->newVersion : p->oldVersion;
					int verHighBits = versionFromPatch & 0xFFFF0000;
					int verLowBits = ((versionFromPatch & 0x0000FFFF) + 1) & 0x0000FFFF;
					HK_ASSERT(0x1deca676, verLowBits != 0);
					return findNewOrReuseVersion(name, verHighBits | verLowBits);
				}
				return version;
			}

			hkDataClass getCurrentClass(const hkClass& klass) const
			{
				hkDataClass currentClass = m_world.findClass(klass.getName());
				HK_ASSERT(0x25d29bea, currentClass.getVersion() == klass.getDescribedVersion());
				return currentClass;
			}

			const hkVersionPatchManager::PatchInfo* getSourcePatchForClass(hkDataClass klass) const
			{
				hkUint64 uid = m_patchManager.getUid(klass.getName(), klass.getVersion());
				return m_srcPatchFromUid.getWithDefault(uid, HK_NULL);
			}

			const hkVersionPatchManager::PatchInfo* getDestinationPatchForClass(hkDataClass klass) const
			{
				hkUint64 uid = m_patchManager.getUid(klass.getName(), klass.getVersion());
				return m_dstPatchFromUid.getWithDefault(uid, HK_NULL);
			}

			const char* getPreviousClassName(const hkDataClass& klass) const
			{
				const hkVersionPatchManager::PatchInfo* dstPatch = getDestinationPatchForClass(klass);
				if( dstPatch && dstPatch->oldName )
				{
					return dstPatch->oldName;
				}
				// no patch found, may be a new class
				return HK_NULL;
			}

			int getPreviousClassVersion(const hkDataClass& klass) const
			{
				const hkVersionPatchManager::PatchInfo* dstPatch = getDestinationPatchForClass(klass);
				if( dstPatch && dstPatch->oldName )
				{
					return dstPatch->oldVersion;
				}
				// no patch found, may be a new class
				return -1;
			}

		private:
			const hkVersionPatchManager& m_patchManager;
			hkDataWorldNative m_world;
			hkDataWorldDict m_manifestWorld;
			mutable hkStringMap<hkDataClassImpl*> m_manifestClassFromName;

			hkPointerMap<hkUint64, const hkVersionPatchManager::PatchInfo*> m_srcPatchFromUid;
			hkPointerMap<hkUint64, const hkVersionPatchManager::PatchInfo*> m_dstPatchFromUid;
	};

	class GenericClassComparator
	{
		public:
			GenericClassComparator(const GenericClassCollector& collector, const hkDataClass& oldClass, const hkDataClass& newClass, const hkClassNameRegistry& classReg) :
				m_collector(collector), m_oldClass(oldClass), m_newClass(newClass), m_classReg(classReg), m_parent(HK_DEPENDENCY_HAS_NO_CHANGES), m_renamed(false), m_shouldHaveFunction(false)
			{
			}

			// return true if class versions are different
			inline hkBool32 foundDifferences() const
			{
				return bool(hasNewParent()) || m_renamed || m_addedMembers.getSize() > 0 || m_removedMembers.getSize() > 0 || m_changedTypeMembers.getSize() || m_changedDefaultMembers.getSize();
			}

			inline const hkDataClass& getOldClass() const
			{
				return m_oldClass;
			}

			inline const hkDataClass& getNewClass() const
			{
				return m_newClass;
			}

			inline const hkArray<const char*>& getAddedMembers() const
			{
				return m_addedMembers;
			}

			inline const hkArray<const char*>& getRemovedMembers() const
			{
				return m_removedMembers;
			}

			inline const hkArray<const char*>& getChangedTypeMembers() const
			{
				return m_changedTypeMembers;
			}

			inline const hkArray<const char*>& getChangedDefaultMembers() const
			{
				return m_changedDefaultMembers;
			}

			inline hkBool32 hasNewParent() const
			{
				return m_parent != HK_DEPENDENCY_HAS_NO_CHANGES;
			}

			inline hkBool32 shouldHaveFunction() const
			{
				return m_shouldHaveFunction;
			}

			inline const DependenciesMap& getOldClassDependencies() const
			{
				return m_oldDependencies;
			}

			inline const DependenciesMap& getNewClassDependencies() const
			{
				return m_newDependencies;
			}

		private:

			inline void checkDependency(const hkDataClass& klass, const hkDataClass::MemberInfo& member, DependenciesMap& dependencies)
			{
				hkDataObject::Type term = member.m_type->findTerminal();
				if (term->isClass() && term->getTypeName() != HK_NULL)
				{
					hkDataClassImpl* memClass = klass.getWorld()->findClass(term->getTypeName());
					HK_ASSERT(0x34324a32, memClass);

					setHierarchyDependencies(klass, hkDataClass(memClass), dependencies);
				}
			}

			// Returns true if either default is not present (as this is not an error)
			hkBool32 defaultsMatch(const void* valuePtr, const char* className, const char* memberName)
			{
				hkTypedUnion attributeValue;
				const hkClass* klass = m_classReg.getClassByName(className);
				HK_ASSERT2(0x26c7956c, klass, "Invalid class name in default check");
				int hkClassIndex = klass->getDeclaredMemberIndexByName(memberName);
				HK_ASSERT2(0xe76a164, hkClassIndex > -1, "Invalid member name in default check");
				hkResult hasAttributeDefault = klass->getDeclaredDefault(hkClassIndex, attributeValue);
				// We need this for enum types to get the underlying type
				const hkClassMember& klassMember = klass->getDeclaredMember(hkClassIndex);
				if(hasAttributeDefault == HK_SUCCESS)
				{
					if(valuePtr)
					{
						// Both the hkClass and the hkDataObject have default values -- check that they are the same
						// The attribute default is only set at the size of the underlying type, not the more abstract
						// patch type (i.e. size of int/real does matter) so we can only compare at that level
						hkClassMember::Type memType = attributeValue.getType();
						switch(memType)
						{
#define CHECK_SIMPLE_TYPE(VALUE_TYPE, STORAGE_TYPE) return !VALUE_TYPE(*reinterpret_cast<const hkInt64*>(valuePtr) != attributeValue.getStorage().m##STORAGE_TYPE)
#define CHECK_REAL_TYPE(VALUE_TYPE, STORAGE_TYPE)  return hkMath::equal(*reinterpret_cast<const VALUE_TYPE*>(valuePtr), attributeValue.getStorage().m##STORAGE_TYPE)
#define CHECK_VEC_TYPE(STORAGE_TYPE, SIZE) for(int vecIndex=0;vecIndex<SIZE;vecIndex++) { if(!hkMath::equal((reinterpret_cast<const float*>(valuePtr))[vecIndex], attributeValue.getStorage().m##STORAGE_TYPE[vecIndex]))\
							{ return false; } } return true
						case hkClassMember::TYPE_INT8:
							CHECK_SIMPLE_TYPE(hkInt8, _int8);
							break;

						case hkClassMember::TYPE_UINT8:
						case hkClassMember::TYPE_BOOL:
							CHECK_SIMPLE_TYPE(hkUint8, _uint8);
							break;
						case hkClassMember::TYPE_INT16:
							CHECK_SIMPLE_TYPE(hkInt16, _int16);
							break;
						case hkClassMember::TYPE_HALF:
							{
								const hkReal v0 = *reinterpret_cast<const hkReal*>(valuePtr);
#if defined(HK_HALF_IS_FLOAT)
								const hkHalf* v1 = reinterpret_cast<const hkHalf*>(&(attributeValue.getStorage().m_uint32));
#else
								const hkHalf* v1 = reinterpret_cast<const hkHalf*>(&(attributeValue.getStorage().m_uint16));
#endif
								return !(v0 != v1->getReal());
							}
						case hkClassMember::TYPE_UINT16:
							CHECK_SIMPLE_TYPE(hkUint16, _uint16);
							break;

						case hkClassMember::TYPE_INT32:
							CHECK_SIMPLE_TYPE(hkInt32, _int32);
							break;
						case hkClassMember::TYPE_UINT32:
#if HK_POINTER_SIZE == 4
						case hkClassMember::TYPE_ULONG:
#endif
							CHECK_SIMPLE_TYPE(hkUint32, _uint32);
							break;
						case hkClassMember::TYPE_INT64:
						case hkClassMember::TYPE_UINT64:
#if HK_POINTER_SIZE == 8
						case hkClassMember::TYPE_ULONG:
#endif
							CHECK_SIMPLE_TYPE(hkInt64, _int64);
							break;

						case hkClassMember::TYPE_ENUM:
							{
								const hkClassMember::Type subType = klassMember.getSubType();
								if((subType == hkClassMember::TYPE_UINT8) || (subType == hkClassMember::TYPE_INT8))
								{
									CHECK_SIMPLE_TYPE(hkInt8, _enumVariant.m_value);
								}
								else if((subType == hkClassMember::TYPE_UINT16) || (subType == hkClassMember::TYPE_INT16))
								{
									CHECK_SIMPLE_TYPE(hkInt16, _enumVariant.m_value);
								}
								else
								{
									CHECK_SIMPLE_TYPE(hkInt32, _enumVariant.m_value);
								}
							}
							break;
						case hkClassMember::TYPE_REAL:
							CHECK_REAL_TYPE(hkReal, _float);
							break;
						case hkClassMember::TYPE_VECTOR4:
						case hkClassMember::TYPE_QUATERNION:
							CHECK_VEC_TYPE(_float4, 4);
							break;
						case hkClassMember::TYPE_QSTRANSFORM:
						case hkClassMember::TYPE_MATRIX3:
							CHECK_VEC_TYPE(_float12, 12);
							break;
						case hkClassMember::TYPE_TRANSFORM:
						case hkClassMember::TYPE_MATRIX4:
							CHECK_VEC_TYPE(_float16, 16);
							break;
						case hkClassMember::TYPE_CSTRING:
						case hkClassMember::TYPE_STRINGPTR:
							{
								const char* firstPtr = *reinterpret_cast<const char* const*>(valuePtr);
								const char* secondPtr = reinterpret_cast<const char*>(attributeValue.getStorage().m_ulong);
								return (!firstPtr && !secondPtr) || !(hkString::strCmp(firstPtr, secondPtr));
							}
						case hkClassMember::TYPE_POINTER:
							// Don't check, we are not going to look into arbitrary pointers
							return true;
						default:
							HK_ASSERT2(0x1d1359ed, 0, "Unsupported type for default patch value");
						}
					}
					return false;
				}
				return true;
			}

		public:
			hkBool32 findClassDifferences()
			{
				DependenciesMap oldOptionalDependencies;
				DependenciesMap newOptionalDependencies;
				// check parent dependencies
				{
					hkDataClass oldParent = m_oldClass.getParent();
					hkDataClass newParent = m_newClass.getParent();
					if( !oldParent.isNull() && newParent.isNull() )
					{
						m_parent = HK_DEPENDENCY_CHANGED;
						m_shouldHaveFunction = true;
						setHierarchyDependencies(m_oldClass, oldParent, oldOptionalDependencies);
					}
					else if( oldParent.isNull() && !newParent.isNull() )
					{
						m_parent = HK_DEPENDENCY_CHANGED;
						m_shouldHaveFunction = true;
						setHierarchyDependencies(m_newClass, newParent, newOptionalDependencies);
					}
					else if( !oldParent.isNull() && !newParent.isNull() )
					{
						if( hkString::strCmp(oldParent.getName(), newParent.getName()) != 0 )
						{
							const char* prevNameOfNewParent = m_collector.getPreviousClassName(newParent); // check renames
							if( !prevNameOfNewParent )
							{
								// new class or renamed parent that has no patch yet
								m_parent = HK_DEPENDENCY_NOT_SURE;
							}
							else if( hkString::strCmp(oldParent.getName(), newParent.getName()) != 0 )
							{
								// new parent, it was not renamed
								m_parent = HK_DEPENDENCY_CHANGED;
								m_shouldHaveFunction = true;
								setHierarchyDependencies(m_oldClass, oldParent, oldOptionalDependencies);
								setHierarchyDependencies(m_newClass, newParent, newOptionalDependencies);
							}
							else
							{
								// a. for some reason existing patch was not applied to upgrade oldParent to newParent
								// b. or patch is missing to update oldClass to previous version of newParent, which must not happen:
								// version verify may assume that only last patch is missing, hkVersionPatchManager::applyPatches() should fail before this.
								HK_ASSERT(0x1593d800, false);
							}
						}
					}
				}
				m_renamed = hkString::strCmp(m_oldClass.getName(), m_newClass.getName()) != 0;

				// check members
				hkArray<hkDataClass::MemberInfo>::Temp oldMinfos(m_oldClass.getNumDeclaredMembers());
				m_oldClass.getAllDeclaredMemberInfo(oldMinfos);
				for( int i = 0; i < oldMinfos.getSize(); ++i )
				{
					const hkDataClass::MemberInfo& oldMinfo = oldMinfos[i];
					const char* memName = oldMinfo.m_name;
					int m;
					if( (m = m_newClass.getDeclaredMemberIndexByName(memName)) != -1 )
					{
						hkDataClass::MemberInfo newMinfo;
						m_newClass.getDeclaredMemberInfo(m, newMinfo);
						if( !HK_DATATYPE_OK(oldMinfo.m_type) && !HK_DATATYPE_OK(newMinfo.m_type) )
						{
							continue;
						}
						if( !HK_DATATYPE_OK(oldMinfo.m_type) && HK_DATATYPE_OK(newMinfo.m_type) )
						{
							m_addedMembers.pushBack(memName);
							m_shouldHaveFunction = m_shouldHaveFunction || hasNewParent() || HK_DATAOBJECT_OK(newMinfo.m_type);
							checkDependency(m_newClass, newMinfo, newOptionalDependencies);
						}
						else if( HK_DATATYPE_OK(oldMinfo.m_type) && !HK_DATATYPE_OK(newMinfo.m_type) )
						{
							m_removedMembers.pushBack(memName);
							m_shouldHaveFunction = m_shouldHaveFunction || hasNewParent() || HK_DATAOBJECT_OK(oldMinfo.m_type);
							checkDependency(m_oldClass, oldMinfo, oldOptionalDependencies);
						}
						else if( !oldMinfo.m_type->isEqual(newMinfo.m_type))
						{
							m_changedTypeMembers.pushBack(memName);
							m_shouldHaveFunction = true;//m_shouldHaveFunction || hasNewParent();
							checkDependency(m_oldClass, oldMinfo, oldOptionalDependencies);
							checkDependency(m_newClass, newMinfo, newOptionalDependencies);
						}

						if(!defaultsMatch(oldMinfo.m_valuePtr, m_newClass.getName(), memName))
						{
							m_changedDefaultMembers.pushBack(memName);
						}
						// members have no differences - happy days
					}
					else
					{
						if( HK_DATATYPE_OK(oldMinfo.m_type) )
						{
							m_removedMembers.pushBack(memName);
							m_shouldHaveFunction = m_shouldHaveFunction || hasNewParent() || HK_DATAOBJECT_OK(oldMinfo.m_type);
							checkDependency(m_oldClass, oldMinfo, oldOptionalDependencies);
						}
					}
				}
				hkArray<hkDataClass::MemberInfo>::Temp newMinfos(m_newClass.getNumDeclaredMembers());
				m_newClass.getAllDeclaredMemberInfo(newMinfos);
				for( int i = 0; i < newMinfos.getSize(); ++i )
				{
					const hkDataClass::MemberInfo& newMinfo = newMinfos[i];
					const char* memName = newMinfo.m_name;
					if( int m = m_oldClass.getDeclaredMemberIndexByName(memName) == -1 )
					{
						if( HK_DATATYPE_OK(newMinfo.m_type) )
						{
							m_addedMembers.pushBack(memName);
							m_shouldHaveFunction = m_shouldHaveFunction || hasNewParent() || HK_DATAOBJECT_OK(newMinfo.m_type);
							checkDependency(m_newClass, newMinfo, newOptionalDependencies);
						}
					}
				}

				setDependencies(m_oldClass, oldOptionalDependencies, m_oldDependencies);
				setDependencies(m_newClass, newOptionalDependencies, m_newDependencies);

				return foundDifferences();
			}

			void setDependencies(const hkDataClass& klass, const DependenciesMap& optional, DependenciesMap& dependenciesOut)
			{
				const hkDataWorld* world = klass.getWorld();
				HK_ASSERT(0x2f0f2278, world);
				for( DependenciesMap::Iterator it = optional.getIterator(); optional.isValid(it); it = optional.getNext(it) )
				{
					hkDataClass dependClass = world->findClass(optional.getKey(it));
					int version = optional.getValue(it);
					HK_ASSERT(0x559841c9, version == dependClass.getVersion());
					if( !localHierarchy(klass, dependClass) )
					{
						dependenciesOut.insert(dependClass.getName(), version);
					}
				}
			}

			const GenericClassCollector& m_collector;
			const hkDataClass& m_oldClass;
			const hkDataClass& m_newClass;
			const hkClassNameRegistry& m_classReg;
			hkEnum<Dependency, hkInt8> m_parent;
			hkBool m_renamed;
			hkBool32 m_shouldHaveFunction;
			hkArray<const char*> m_addedMembers;
			hkArray<const char*> m_removedMembers;
			hkArray<const char*> m_changedTypeMembers;
			hkArray<const char*> m_changedDefaultMembers;
			DependenciesMap m_oldDependencies;
			DependenciesMap m_newDependencies;
	};

	static inline void reportMemberRenamed(hkOstream& output, const hkDataClass::MemberInfo& m0, const hkDataClass::MemberInfo& m1)
	{
		output << "\t" << sPatchMemberRenamed << "(\"" << m0.m_name << "\", \"" << m1.m_name << "\")\n";
	}

	static inline void reportMemberRemoved(hkOstream& output, const hkDataClass& klass, const hkDataClass::MemberInfo& m)
	{
		hkStringBuf typeBuf;
		int tupleSize;
		const char* typeString = getStringFromType(m.m_type, typeBuf, tupleSize);
		hkStringBuf buf;
		output << "\t" << sPatchMemberRemoved << "(\"" << m.m_name << "\", " << typeString << ", " << REPORT_TYPE_NAME(m, buf) << ", " << tupleSize << ")\n";
	}

	static inline void reportMemberDefaultSet(hkOstream& output, const hkDataClass& klass, const hkDataClass::MemberInfo& m, const hkClassNameRegistry& classReg)
	{
		hkStringBuf buf;
		if(const hkClass* worldClass = classReg.getClassByName(klass.getName()))
		{
			int memberIndex;
			if((memberIndex = worldClass->getDeclaredMemberIndexByName(m.m_name)) != -1)
			{
				if(worldClass->hasDeclaredDefault(memberIndex))
				{
					hkTypedUnion defaultVal;
					worldClass->getDeclaredDefault(memberIndex, defaultVal);
					const hkClassMember& member = worldClass->getDeclaredMember(memberIndex);
					hkClassMember::Type memberType = member.getType();

					if (memberType == hkClassMember::TYPE_FLAGS)
					{
						memberType = member.getSubType();
					}

					switch(memberType)
					{
					case hkClassMember::TYPE_BOOL:
						output << "\t" << sPatchMemberDefaultSetByte << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_ENUM:
					case hkClassMember::TYPE_INT8:
					case hkClassMember::TYPE_UINT8:
					case hkClassMember::TYPE_INT16:
					case hkClassMember::TYPE_UINT16:
					case hkClassMember::TYPE_INT32:
					case hkClassMember::TYPE_UINT32:
					case hkClassMember::TYPE_INT64:
					case hkClassMember::TYPE_UINT64:
					case hkClassMember::TYPE_ULONG:
						output << "\t" << sPatchMemberDefaultSetInt << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_REAL:
					case hkClassMember::TYPE_HALF:
						output << "\t" << sPatchMemberDefaultSetReal << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_VECTOR4:
					case hkClassMember::TYPE_QUATERNION:
						output << "\t" << sPatchMemberDefaultSetVec4 << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_QSTRANSFORM:
					case hkClassMember::TYPE_MATRIX3:
						output << "\t" << sPatchMemberDefaultSetVec12 << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_TRANSFORM:
					case hkClassMember::TYPE_MATRIX4:
						output << "\t" << sPatchMemberDefaultSetVec16 << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_CSTRING:
					case hkClassMember::TYPE_STRINGPTR:
						output << "\t" << sPatchMemberDefaultSetCString << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_POINTER:
						output << "\t" << sPatchMemberDefaultSetPointer << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					default:
						HK_ASSERT2(0x40a9dcc3, 0, "Invalid patch default");
					}
				}
			}
		}
	}

	static inline void reportMemberAdded(hkOstream& output, const hkDataClass& klass, const hkDataClass::MemberInfo& m, const hkClassNameRegistry& classReg)
	{
		hkStringBuf typeBuf;
		int tupleSize;
		const char* typeString = getStringFromType(m.m_type, typeBuf, tupleSize);
		hkStringBuf buf, buf2;
		if(const hkClass* worldClass = classReg.getClassByName(klass.getName()))
		{
			int memberIndex = worldClass->getDeclaredMemberIndexByName(m.m_name);
			if(memberIndex > -1)
			{
				if(worldClass->hasDeclaredDefault(memberIndex))
				{
					hkTypedUnion defaultVal;
					worldClass->getDeclaredDefault(memberIndex, defaultVal);
					const hkClassMember& member = worldClass->getDeclaredMember(memberIndex);
					hkClassMember::Type memberType = member.getType();
					
					if (memberType == hkClassMember::TYPE_FLAGS)
					{
						memberType = member.getSubType();
					}

					switch(memberType)
					{
					case hkClassMember::TYPE_BOOL:
						output << "\t" << sPatchMemberAddedByte << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_ENUM:
					case hkClassMember::TYPE_INT8:
					case hkClassMember::TYPE_UINT8:
					case hkClassMember::TYPE_INT16:
					case hkClassMember::TYPE_UINT16:
					case hkClassMember::TYPE_INT32:
					case hkClassMember::TYPE_UINT32:
					case hkClassMember::TYPE_INT64:
					case hkClassMember::TYPE_UINT64:
					case hkClassMember::TYPE_ULONG:
						output << "\t" << sPatchMemberAddedInt << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_REAL:
					case hkClassMember::TYPE_HALF:
						output << "\t" << sPatchMemberAddedReal << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_VECTOR4:
					case hkClassMember::TYPE_QUATERNION:
						output << "\t" << sPatchMemberAddedVec4 << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_QSTRANSFORM:
					case hkClassMember::TYPE_MATRIX3:
						output << "\t" << sPatchMemberAddedVec12 << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_TRANSFORM:
					case hkClassMember::TYPE_MATRIX4:
						output << "\t" << sPatchMemberAddedVec16 << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_CSTRING:
					case hkClassMember::TYPE_STRINGPTR:
						output << "\t" << sPatchMemberAddedCString << "(\"" << m.m_name << "\", " << REPORT_TYPE_VAL(defaultVal, memberType, buf) << ")\n"; return;
					case hkClassMember::TYPE_POINTER:
						output << "\t" << sPatchMemberAddedPointer << "(\"" << m.m_name << "\", " << REPORT_TYPE_NAME(m, buf) << ", " << REPORT_TYPE_VAL(defaultVal, memberType, buf2) << ")\n"; return;
					default:
						HK_ASSERT2(0x40a9dcc3, 0, "Invalid patch default");
					}

				}				
			}
		}
		output << "\t" << sPatchMemberAdded << "(\"" << m.m_name << "\", " << typeString << ", " << REPORT_TYPE_NAME(m, buf) << ", " << tupleSize << ")\n";
	}

	static inline void reportParentChanged(hkOstream& output, const hkDataClass& oldParent, const hkDataClass& newParent)
	{
		hkStringBuf buf0;
		hkStringBuf buf1;
		output << "\t" << sPatchParentSet << "(" << REPORT_CLASS_NAME(oldParent,buf0) << ", " << REPORT_CLASS_NAME(newParent,buf1) << ")\n";
	}

	static inline void reportFunction(hkOstream& output, const hkDataClass& oldClass, const hkDataClass& newClass, int suggestedVersion)
	{
		hkStringBuf oldVersion; getVersionString(oldClass.getVersion(), oldVersion);
		hkStringBuf newVersion; getVersionString(suggestedVersion, newVersion);
		oldVersion.replace(" ",""); oldVersion.replace("|","_");
		newVersion.replace(" ",""); newVersion.replace("|","_");
		output << "\t//HK_PATCH_FUNCTION(" << oldClass.getName() << "_" << oldVersion << "_to_"  << newVersion << ")\n";
	}

	static inline void reportDependency(hkOstream& output, const char* typeName, int version)
	{
		output << "\t" << sPatchDepends << "(\"" << typeName << "\", " << STREAM_PATCH_VERSION(version) << ")\n";
	}

	static inline void reportGroup(hkOstream& output, const char* typeName, int version)
	{
		output << "\t" << sPatchGroup << "(\"" << typeName << "\", " << STREAM_PATCH_VERSION(version) << ")\n";
	}

	static inline void reportDependencies(hkOstream& output, const GenericClassCollector& collector, const GenericClassComparator& comparator)
	{
		const DependenciesMap& oldDependencies = comparator.getOldClassDependencies();
		const DependenciesMap& newDependencies = comparator.getNewClassDependencies();
		hkLocalArray<DependencyInfo> cachedNewDependencies(10);
		for( DependenciesMap::Iterator it = newDependencies.getIterator(); newDependencies.isValid(it); it = newDependencies.getNext(it) )
		{
			const char* typeName = newDependencies.getKey(it);
			int version = newDependencies.getValue(it);
			DependencyInfo info = { typeName, version };
			cachedNewDependencies.pushBack(info);
		}

		hkStringMap<const char*> renames;
		{
			const hkDataWorld* oldWorld = comparator.getOldClass().getWorld();
			for( DependenciesMap::Iterator it = oldDependencies.getIterator(); oldDependencies.isValid(it); it = oldDependencies.getNext(it) )
			{
				const char* typeName = oldDependencies.getKey(it);
				hkDataClass oldClass = oldWorld->findClass(typeName);
				const hkVersionPatchManager::PatchInfo* srcpatch = collector.getSourcePatchForClass(oldClass);
				if( srcpatch && srcpatch->newName && hkString::strCmp(srcpatch->oldName, srcpatch->newName) != 0 )
				{
					renames.insert(srcpatch->oldName, srcpatch->newName);
				}
			}
		}

		for( DependenciesMap::Iterator it = oldDependencies.getIterator(); oldDependencies.isValid(it); it = oldDependencies.getNext(it) )
		{
			const char* typeName = oldDependencies.getKey(it);
			int oldVersion = oldDependencies.getValue(it);
			const char* newDependencyTypeName = typeName;
			int newVersion = newDependencies.getWithDefault(newDependencyTypeName, -1);
			if( newVersion == -1 )
			{
				newDependencyTypeName = renames.getWithDefault(typeName, typeName); // check renames
				newVersion = newDependencies.getWithDefault(newDependencyTypeName, -1);
			}
			if( newVersion == -1 )
			{
				// no conflicting new dependecies
				reportDependency(output, typeName, oldVersion);
				continue;
			}
			else if( newVersion != oldVersion )
			{
				// the same class, but different versions
				reportGroup(output, typeName, oldVersion); // conflict of class version dependencies
			}
			else
			{
				reportDependency(output, typeName, oldVersion);
			}
			{
				DependencyInfo info = { newDependencyTypeName, newVersion };
				int idx = cachedNewDependencies.indexOf(info);
				HK_ASSERT(0x2a9449e5, idx != -1);
				cachedNewDependencies.removeAt(idx);
			}
		}
		for( int i = 0; i < cachedNewDependencies.getSize(); ++i )
		{
			reportDependency(output, cachedNewDependencies[i].typeName, cachedNewDependencies[i].version);
		}
	}

	static void reportNewClass(hkOstream& report, const hkDataClass& klass, int suggestedVersion, const GenericClassCollector& collector, const hkClassNameRegistry& classReg)
	{
		hkStringBuf buf;
		report << "\nHK_PATCH_BEGIN(HK_NULL, HK_CLASS_ADDED, \"" << klass.getName() << "\", " << HK_VER(suggestedVersion,buf) << ")\n";
		DependenciesMap dependencies;
		if( !klass.getParent().isNull() )
		{
			hkDataClass oldParent(HK_NULL);
			reportParentChanged(report, oldParent, klass.getParent());
			setHierarchyDependencies(klass, klass.getParent(), dependencies);
		}
		hkArray<hkDataClass::MemberInfo>::Temp members(klass.getNumDeclaredMembers());
		klass.getAllDeclaredMemberInfo(members);
		for( int i = 0; i < members.getSize(); ++i )
		{
			hkDataClass::MemberInfo& m = members[i];
			if( HK_DATATYPE_OK(m.m_type) )
			{
				reportMemberAdded(report, klass, m, classReg);
				hkDataObject::Type term = m.m_type->findTerminal();

				if( term->isClass())
				{
					hkDataClassImpl* cls = klass.getWorld()->findClass(term->getTypeName());
					HK_ASSERT(0x32423432, cls);

					hkDataClass dependClass(cls);
					if( !localHierarchy(klass, dependClass) )
					{
						setHierarchyDependencies(klass, dependClass, dependencies);
					}
				}
			}
		}
		for( DependenciesMap::Iterator it = dependencies.getIterator(); dependencies.isValid(it); it = dependencies.getNext(it) )
		{
			reportDependency(report, dependencies.getKey(it), dependencies.getValue(it));
		}
		report << "HK_PATCH_END()\n";
	}

	static void reportRemovedClass(hkOstream& report, const hkDataClass& klass, const GenericClassCollector& collector)
	{
		hkStringBuf buf;
		report << "\nHK_PATCH_BEGIN(\"" << klass.getName() << "\", " << HK_VER(klass.getVersion(),buf) << ", HK_NULL, HK_CLASS_REMOVED)\n";
		DependenciesMap dependencies;
		if( !klass.getParent().isNull() )
		{
			hkDataClass nullParent(HK_NULL);
			reportParentChanged(report, klass.getParent(), nullParent);
			setHierarchyDependencies(klass, klass.getParent(), dependencies);
		}
		hkArray<hkDataClass::MemberInfo>::Temp members(klass.getNumDeclaredMembers());
		klass.getAllDeclaredMemberInfo(members);
		for( int i = 0; i < members.getSize(); ++i )
		{
			hkDataClass::MemberInfo& m = members[i];
			if( HK_DATATYPE_OK(m.m_type) )
			{
				reportMemberRemoved(report, klass, m);

				hkDataObject::Type term = m.m_type->findTerminal();
				if (term->isClass())
				{
					const hkDataClassImpl* dependImpl = klass.getWorld()->findClass(term->getTypeName());
					HK_ASSERT(0x23423432, dependImpl);

					hkDataClass dependClass(const_cast<hkDataClassImpl*>(dependImpl));
					if( !localHierarchy(klass, dependClass) )
					{
						setHierarchyDependencies(klass, dependClass, dependencies);
					}
				}
			}
		}
		for( DependenciesMap::Iterator it = dependencies.getIterator(); dependencies.isValid(it); it = dependencies.getNext(it) )
		{
			reportDependency(report, dependencies.getKey(it), dependencies.getValue(it));
		}
		report << "HK_PATCH_END()\n";
	}

	static void reportClassChanges(hkOstream& report, const hkDataClass& oldClass, const hkDataClass& newClass, int suggestedVersion, const GenericClassCollector& collector, const GenericClassComparator& comparator, const hkClassNameRegistry& classReg)
	{
		hkStringBuf buf0;
		hkStringBuf buf1;
		report << "\nHK_PATCH_BEGIN(\"" << oldClass.getName() << "\", " << HK_VER(oldClass.getVersion(),buf0) << ", \"" << newClass.getName() << "\", " << HK_VER(suggestedVersion,buf1) << ")\n";
		const char* oldParentName = !oldClass.getParent().isNull() ? oldClass.getParent().getName() : HK_NULL;
		const char* newParentName = !newClass.getParent().isNull() ? newClass.getParent().getName() : HK_NULL;
		// check renames here ???
		// ...

		if( (!oldParentName && newParentName)
			|| (oldParentName && !newParentName)
			|| (oldParentName && newParentName && hkString::strCmp(oldParentName, newParentName) != 0) )
		{
			reportParentChanged(report, oldClass.getParent(), newClass.getParent());
		}

		const hkArray<const char*>& changedMembers = comparator.getChangedTypeMembers();
		for( int i = 0; i < changedMembers.getSize(); ++i )
		{
			hkDataClass::MemberInfo minfo;
			oldClass.getDeclaredMemberInfo(oldClass.getDeclaredMemberIndexByName(changedMembers[i]), minfo);
			reportMemberRemoved(report, oldClass, minfo);
			newClass.getDeclaredMemberInfo(newClass.getDeclaredMemberIndexByName(changedMembers[i]), minfo);
			reportMemberAdded(report, newClass, minfo, classReg);
		}
		const hkArray<const char*>& changedDefaultMembers = comparator.getChangedDefaultMembers();
		for( int i = 0; i < changedDefaultMembers.getSize(); ++i )
		{
			hkDataClass::MemberInfo minfo;
			newClass.getDeclaredMemberInfo(newClass.getDeclaredMemberIndexByName(changedDefaultMembers[i]), minfo);
			reportMemberDefaultSet(report, newClass, minfo, classReg);
		}

		const hkArray<const char*>& removedMembers = comparator.getRemovedMembers();
		for( int i = 0; i < removedMembers.getSize(); ++i )
		{
			hkDataClass::MemberInfo minfo;
			oldClass.getDeclaredMemberInfo(oldClass.getDeclaredMemberIndexByName(removedMembers[i]), minfo);
			reportMemberRemoved(report, oldClass, minfo);
		}
		const hkArray<const char*>& addedMembers = comparator.getAddedMembers();
		for( int i = 0; i < addedMembers.getSize(); ++i )
		{
			hkDataClass::MemberInfo minfo;
			newClass.getDeclaredMemberInfo(newClass.getDeclaredMemberIndexByName(addedMembers[i]), minfo);
			reportMemberAdded(report, newClass, minfo, classReg);
		}
		reportDependencies(report, collector, comparator);
		if( comparator.shouldHaveFunction() )
		{
			reportFunction(report, oldClass, newClass, suggestedVersion);
		}
		report << "HK_PATCH_END()\n";
	}

	static void reportWrongVersion(hkOstream& report, const hkDataClass& oldClass, const hkDataClass& newClass, int suggestedVersion, const GenericClassCollector& collector, const GenericClassComparator& comparator)
	{
		report << newClass.getName() << "\t// +version(" << suggestedVersion << ")\n";
	}
}

hkResult HK_CALL hkVersionCheckingUtils::verifyClassPatches(hkOstream& report, const hkDataWorld& world, const hkClassNameRegistry& classReg, const hkVersionPatchManager& patchManager, Flags flags)
{
	hkResult result = HK_SUCCESS;
	hkArray<char> bufOfChanges;
	hkOstream reportChanges(bufOfChanges);
	hkArray<char> bufOfFixVer;
	hkOstream reportFixVer(bufOfFixVer);
	hkArray<char> bufOfClasses;
	hkOstream reportClasses(bufOfClasses);

	GenericClassCollector collector(world, classReg, patchManager);

	hkArray<const hkClass*> classes;
	classReg.getClasses(classes);
	hkStringBuf buf;
	for( int i = 0; i < classes.getSize(); ++i )
	{
		if( classes[i]->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
		{
			continue;
		}

		////////////////////////////////////////////////////////////////////////
		///////////////////////////////// HACK /////////////////////////////////
		/////////////////// REMOVE WHEN RELEASING OPENFLIGHT  //////////////////
		////////////////////////////////////////////////////////////////////////
		if( hkString::strNcmp(classes[i]->getName(), "hkmsFlt", 4) == 0)
		{
			// skip hkmsFlt classes until ready for release
			collector.getAndMarkManifestClass(classes[i]->getName());
			continue;
		}
		////////////////////////////////////////////////////////////////////////
		/////////////////////////////// END HACK ///////////////////////////////
		////////////////////////////////////////////////////////////////////////

		#if HAVOK_BUILD_NUMBER == 0 // special case negative version numbers as "don't care about versioning" for internal builds
			if( classes[i]->getDescribedVersion() < 0 )
			{
				reportChanges << "Class \"" << classes[i]->getName() << "\" is marked as unstable, and will not be checked for patches\n";
				collector.getAndMarkManifestClass(classes[i]->getName());
				continue;
			}
		#endif
		hkDataClass klass = collector.getCurrentClass(*classes[i]);
		const hkVersionPatchManager::PatchInfo* dstNewPatch = collector.getDestinationPatchForClass(klass);
		hkDataClass oldClass = collector.getAndMarkManifestClass(klass.getName());
		if( !oldClass.isNull() )
		{
			bool sameVersionNum = klass.getVersion() == oldClass.getVersion();
			int suggestedVersion = collector.findNewOrReuseVersion(klass.getName(), klass.getVersion());
			GenericClassComparator classComparator(collector, oldClass, klass, classReg);

			if( classComparator.findClassDifferences() )
			{
				result = HK_FAILURE;
				if( sameVersionNum )
				{
					if( !(flags & VERBOSE) )
					{
						reportChanges << "Class \"" << klass.getName() << "\" is changed, the class version number " << HK_VER(oldClass.getVersion(),buf);
						reportChanges << " is not updated (" << HK_VER(suggestedVersion,buf) << ").\n";
					}
				}
				else
				{
					if( !(flags & VERBOSE) )
					{
						reportChanges << "Class \"" << klass.getName() << "\" (" << HK_VER(oldClass.getVersion(),buf);
						reportChanges << ", " << HK_VER(klass.getVersion(),buf);
						reportChanges << ") is changed.\n\tMake sure the class version matches version used in patch (" << HK_VER(klass.getVersion(),buf) << ").\n";
					}
				}
				if( flags & VERBOSE )
				{
					reportClassChanges(reportChanges, oldClass, klass, suggestedVersion, collector, classComparator, classReg);
				}
			}
			else
			{
				if( !sameVersionNum )
				{
					result = HK_FAILURE;
					if( flags & VERBOSE )
					{
						reportWrongVersion(reportFixVer, oldClass, klass, oldClass.getVersion(), collector, classComparator);
					}
					else
					{
						reportFixVer << "Class \"" << klass.getName() << "\" versions are different (" << HK_VER(oldClass.getVersion(),buf);
						reportFixVer << ", " << HK_VER(klass.getVersion(),buf);
						reportFixVer << "). Version of unchanged class must match version of the last class patch (" << HK_VER(oldClass.getVersion(),buf) << ").\n";
					}
				}
			}
		}
		else
		{
			if( hkString::beginsWith(klass.getName(), "hkValueTypeFreeListhkp" ) ) continue;
			int suggestedVersion = collector.findNewOrReuseVersion(klass.getName(), klass.getVersion());
			if( !dstNewPatch )
			{
				result = HK_FAILURE;
				if( flags & VERBOSE )
				{
					reportNewClass(reportClasses, klass, suggestedVersion, collector, classReg);
				}
				else
				{
					reportClasses << "Class \"" << klass.getName() << "\" is new (" << HK_VER(suggestedVersion,buf) << ").\n";
					if( suggestedVersion != klass.getVersion() )
					{
						reportClasses << "\t\"" << klass.getName() << "\" class version is set to " << HK_VER(klass.getVersion(),buf);
						reportClasses << ", but should be " << HK_VER(suggestedVersion,buf) << ".\n";
					}
				}
			}
			else
			{
				result = HK_FAILURE;
				if( flags & VERBOSE )
				{
					reportNewClass(reportClasses, klass, suggestedVersion, collector, classReg);
				}
				else
				{
					reportClasses << "Class \"" << klass.getName() << "\" is new (" << HK_VER(suggestedVersion,buf) << ").\n";
					if( suggestedVersion != klass.getVersion() )
					{
						reportClasses << "\t\"" << klass.getName() << "\" class version is set to " << HK_VER(klass.getVersion(),buf);
						reportClasses << ", but should be " << HK_VER(suggestedVersion,buf) << ".\n";
					}
				}
			}
		}
	}
	hkArray<hkDataClassImpl*> removedClasses;
	collector.getUnmarkedManifestClasses(removedClasses);
	for( int i = 0; i < removedClasses.getSize(); ++i )
	{
		result = HK_FAILURE;
		hkDataClass klass(removedClasses[i]);
		// report removed classes
		if( flags & VERBOSE )
		{
			reportRemovedClass(reportClasses, klass, collector);
		}
		else
		{
			reportClasses << "Class \"" << klass.getName() << "\" is removed (" << HK_VER(klass.getVersion(),buf) << ").\n";
		}
	}

	bool shouldFixAlreadyFoundProblemsFirst = false;
	if( bufOfFixVer.getSize() > 0 ) // reportFixVer
	{
		report.write(bufOfFixVer.begin(), bufOfFixVer.getSize());
		shouldFixAlreadyFoundProblemsFirst = true;
	}
	if( bufOfClasses.getSize() > 0 ) // reportClasses
	{
		if( shouldFixAlreadyFoundProblemsFirst )
		{
			report << "\nIMPORTANT! There are more patch issues found. Please fix already reported patch problems first.\n";
			return result;
		}
		report.write(bufOfClasses.begin(), bufOfClasses.getSize());
		shouldFixAlreadyFoundProblemsFirst = true;
	}
	if( bufOfChanges.getSize() > 0 ) // reportChanges
	{
		if( shouldFixAlreadyFoundProblemsFirst )
		{
			report << "\nIMPORTANT! There are more patch issues found. Please fix already reported patch problems first.\n";
			return result;
		}
		report.write(bufOfChanges.begin(), bufOfChanges.getSize());
	}
	return result;
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
