/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>

#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Types/hkTypedUnion.h>

#include <Common/Base/Reflection/Util/hkVariantDataUtil.h>

#if 0
#	include <Common/Base/Fwd/hkcstdio.h>
#	define TRACE(A) printf A
#else
#	define TRACE(A)
#endif

static const char* s_hkDataObjectTypeAttributeID = "hk.DataObjectType";
static hkDataArrayImpl* hkDataArrayDict_create( hkDataWorldDict* world, hkDataObject::Type t, int size = 0);

static hkReal* hkRealArrayImplementation_getReals(hkDataArrayImpl* array, int numReals);


// We rely on this when allocating and freeing hkDataRefCounted
HK_COMPILE_TIME_ASSERT(HK_OFFSET_OF(hkReferencedObject, m_memSizeAndFlags) == HK_OFFSET_OF(hkDataRefCounted, m_memSize));

class hkDataObjectDict;

namespace
{
	struct InternedString
	{
		InternedString() : m_cachedString(HK_NULL) {}

		inline hkBool32 operator ==( const InternedString& o ) const { return m_cachedString == o.m_cachedString; }
		inline hkBool32 operator !=( const InternedString& o ) const { return m_cachedString != o.m_cachedString; }

		inline const char* getString() const { return m_cachedString; }

		friend class ::hkDataObjectDict;
		friend class ::hkDataWorldDict::ObjectTracker;

		explicit InternedString(const char* s) : m_cachedString(s) {}
	private:

		const char* m_cachedString;
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, InternedString);
	};

	class InternedStringRefCounted : public hkReferencedObject
	{
	public:
		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE);

		InternedStringRefCounted() : m_cachedString(HK_NULL)
		{
			m_memSizeAndFlags = hkSizeOf(InternedStringRefCounted);
		}

		InternedStringRefCounted(const InternedString& s) : m_cachedString(s.getString())
		{
		}

		inline hkBool32 operator ==( const InternedStringRefCounted& o ) const { return m_cachedString == o.m_cachedString; }
		inline hkBool32 operator !=( const InternedStringRefCounted& o ) const { return m_cachedString != o.m_cachedString; }

		inline const char* getString() const { return m_cachedString; }

		friend class ::hkDataObjectDict;
		friend class ::hkDataWorldDict::ObjectTracker;

	private:
		const char* m_cachedString;
	};
	static InternedStringRefCounted InternedNullRefCounted;

	struct InternedStringHandle
	{
		inline const char* getString() const { return m_cached->getString(); }
		InternedStringHandle() { m_cached = &InternedNullRefCounted; }
		inline hkBool32 operator ==( const InternedStringHandle& o ) const { return m_cached == o.m_cached; }
		inline hkBool32 operator !=( const InternedStringHandle& o ) const { return m_cached != o.m_cached; }

		inline void clear() { m_cached = &InternedNullRefCounted; }
		friend class ::hkDataWorldDict::ObjectTracker;

	private:
		hkRefPtr<InternedStringRefCounted> m_cached;
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, InternedStringHandle);
	};

	union Union
	{
		hkInt64 i;
		//		hkUlong u;
		void* p;
		hkReal r;
		char* s;
		hkDataObjectImpl* o;
		hkDataArrayImpl* a;
		//hkReal* ra;

		static void clearArray( hkDataObject::Type typeIn, Union* u, int n, hkBool32 destroyStructs)
		{
			if( typeIn->isArray() || typeIn->isTuple())
			{
				HK_ASSERT(0x12a4423, n == 1);

				hkDataArrayImpl* array = u->a;

				if (typeIn->getParent()->isClass())
				{
					// It's an array of structs

				}

				if (array)
				{
					array->removeReference();
				}
				u->a = HK_NULL;
			}
			else
			{
				switch(typeIn->getSubType())
				{
					case hkTypeManager::SUB_TYPE_VOID:
					case hkTypeManager::SUB_TYPE_BYTE: // used only in arrays
					case hkTypeManager::SUB_TYPE_INT:
					case hkTypeManager::SUB_TYPE_REAL:
						break;
					case hkTypeManager::SUB_TYPE_POINTER: 
					{
						for( int i = 0; i < n; ++i )
						{
							hkDataObjectImpl* obj =  u[i].o;
							u[i].o = HK_NULL;
							if(obj )
							{
								static_cast<hkDataRefCounted*>(obj)->removeReference();
							}
						}
						break;
					}
					case hkTypeManager::SUB_TYPE_CLASS:
					{
						for( int i = 0; i < n; ++i )
						{
							hkDataObjectImpl* obj =  u[i].o;
							u[i].o = HK_NULL;
							if(obj )
							{
								if (destroyStructs)
								{
									obj->destroy();
								}
								static_cast<hkDataRefCounted*>(obj)->removeReference();
							}
						}
						break;
					}
					case hkTypeManager::SUB_TYPE_CSTRING:
					{
						for( int i = 0; i < n; ++i )
						{
							hkString::strFree( u[i].s );
							u[i].s = 0;
						}
						break;
					}
					default:
						HK_ASSERT(0x11c59840, 0);
				}
			}
		}

		void clear(hkDataObject::Type typeIn, hkBool32 destroyStructs = false)
		{
			clearArray( typeIn, this, 1, destroyStructs );
		}
	};

	static hkBool32 typesCompatible( hkDataObject::Type memberType, hkTypeManager::SubType subType)
	{
		// If the subType is void means we don't care what type the underlying member is
		if (subType == hkTypeManager::SUB_TYPE_VOID)
		{
			return true;
		}

		HK_ASSERT(0x2813f9d1, !memberType->isVoid());

		// If they are the same they are 'compatible'. 
		// There may be a problem with pointers... or array types, but that'd have to be checked elsewhere.
		if (memberType->getSubType() == subType)
		{
			return true;
		}

		// Arrays/tuples are interchangable
		if (subType == hkTypeManager::SUB_TYPE_ARRAY)
		{
			return memberType->getSubType() == hkTypeManager::SUB_TYPE_TUPLE;
		}
		if (subType == hkTypeManager::SUB_TYPE_TUPLE)
		{
			return memberType->getSubType() == hkTypeManager::SUB_TYPE_ARRAY;
		}

	
		if (subType == hkTypeManager::SUB_TYPE_CLASS)
		{
			// Can be a pointer to a class to
			return (memberType->getSubType() == hkTypeManager::SUB_TYPE_POINTER) &&
				(memberType->getParent()->getSubType() == hkTypeManager::SUB_TYPE_CLASS);
		}

		return false;
	}
}

//////////////////////////////////////////////////////////////////////////
// WorldDict (hkDataWorldDict::ObjectTracker)
//////////////////////////////////////////////////////////////////////////

class hkDataWorldDict::ObjectTracker
{
public:
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, hkDataWorldDict::ObjectTracker);

	ObjectTracker(hkMemoryAllocator* mall);
	~ObjectTracker();
    typedef hkSerializeMultiMap<hkDataClassDict*, hkDataObjectDict*> ObjectsFromClassMap;
    typedef hkSerializeMultiMap<hkDataClassDict*, hkDataClassDict*> DerivedClassFromParentMap;

	//inline hkMemoryAllocator& getAllocator() const { return *m_allocator; }
	inline void trackClass(hkDataClassDict* klass);
	inline void untrackClass(hkDataClassDict* klass);
	inline hkDataClassDict* findTrackedClass(const char* name) const
	{
		return name ? m_classes.getWithDefault(name, HK_NULL) : HK_NULL;
	}
	inline void findTrackedClasses(hkArray<hkDataClassImpl*>::Temp& classesOut) const;
	inline void retrackRenamedClass(const char* oldName, const char* newName);
	inline void retrackDerivedClass(hkDataClassDict* oldParent, hkDataClassDict* klass);

	inline void trackStructArray(hkDataArrayImpl* a);
	inline void untrackStructArray(hkDataArrayImpl* a);
	inline void getTrackedStructArrays(hkDataClassDict* c, hkBool32 baseClass, hkArray<hkDataArrayImpl*>::Temp& arraysOut);

	inline void trackObject(hkDataObjectDict* obj);
	inline void untrackObject(hkDataObjectDict* obj);
	inline hkDataObjectDict* getTopLevelObject();
	inline void getTrackedObjects(const char* className, hkBool32 baseClass, hkBool32 addStructs, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const;
	inline void retractCastedObject(hkDataObjectDict* obj, hkDataClassDict* newClass);
	inline void retractCastedObjects(hkDataClassDict* oldClass, hkDataClassDict* newClass);

	inline InternedString intern(const char*);
	inline InternedStringHandle internClassNameHandle(const char*);

private:
	inline void replaceClassNameInInternHandle(const char* oldName, const char* newName);
	inline void removeClassContent(hkDataClassDict* klass);
	inline void findTrackedObjectsByBase(const char* className, hkBool32 baseClass, hkBool32 addStructs, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const;

	hkMemoryAllocator* m_allocator;
	hkStringMap<hkDataClassDict*> m_classes; // classes map (primary)
	DerivedClassFromParentMap m_derivedFromParent; //
	hkDataObjectDict* m_topLevelObject;
	ObjectsFromClassMap m_objectsFromClass; // objects from class name map (must be synced to classes and objects)
	hkStringMap<char*> m_interns;
	hkStringMap<InternedStringRefCounted*> m_internedClassNames;

	typedef hkPointerMap<hkDataArrayImpl*, int> ArrayMap;
	typedef hkPointerMap<hkDataClassDict*, ArrayMap*> ClassToArraysMap;

	ClassToArraysMap m_arraysFromClass;
};

////////////////////////////////////////////////////////////////////////
// Class
//////////////////////////////////////////////////////////////////////////

class hkDataClassDict : public hkDataClassImpl
{
	public:

		struct MemberInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, MemberInfo);

			MemberInfo() : m_type(HK_NULL)
			{
				m_default.i = 0;
			}
			MemberInfo(const MemberInfo& other)
				: m_name(other.m_name), m_default(other.m_default), m_type(other.m_type), m_valuePtr(other.m_valuePtr)
			{
			}

			void operator=(const MemberInfo& other)
			{
				m_name = other.m_name;
				m_default = other.m_default;
				m_type = other.m_type;
				m_valuePtr = other.m_valuePtr;
			}
			void getMemberInfo(const hkDataClassImpl* cls, hkDataClass::MemberInfo& info) const
			{
				info.m_name = m_name.getString();
				info.m_owner = cls;
				info.m_type = m_type;
				info.m_valuePtr = m_valuePtr;
			}

			InternedString m_name;
			Union m_default;
			hkDataObject::Type m_type;
			const void* m_valuePtr;
		};

		struct Enum
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, Enum);
			struct Item
			{
				InternedString name;
				int value;
			};
			InternedString m_name;
			hkArrayBase<Item> m_items;
		};

		hkDataWorldDict* m_world;
		InternedStringHandle m_name;
		int m_version;
		hkRefPtr<hkDataClassDict> m_parent;
		hkArrayBase<Enum*> m_enums;
		hkArrayBase<MemberInfo> m_memberInfo;
		hkArrayBase<MemberInfo> m_hiddenMemberInfo;

		inline hkMemoryAllocator& getAllocator() const
		{
			return *m_world->m_allocator;
		}

		// TODO add method to check that all strings are interned
		inline InternedString intern(const char* in) const
		{
			return m_world->m_tracker->intern(in);
		}

		inline InternedStringHandle internClassNameHandle(const char* in) const
		{
			HK_ASSERT(0x234143a8, m_world);
			return m_world->m_tracker->internClassNameHandle(in);
		}

		hkDataClassDict( hkDataWorldDict* root, const char* name, int version )
			: m_world(root)
			, m_name( internClassNameHandle(name) )
			, m_version(version)
			, m_parent(HK_NULL)
		{
		}

		hkDataClassDict( hkDataWorldDict* root, const hkClass& klass, hkDataClassDict* parent, int overrideVersion = -1 )
			: m_world(root)
			, m_name( internClassNameHandle(klass.getName()) )
			, m_version(overrideVersion!=-1 ? overrideVersion : klass.getDescribedVersion() )
			, m_parent(parent)
		{
			hkMemoryAllocator& allocator = getAllocator();
			m_enums._setSize(allocator, klass.getNumDeclaredEnums());
			for( int enumIndex = 0; enumIndex < klass.getNumDeclaredEnums(); ++enumIndex )
			{
				const hkClassEnum& esrc = klass.getDeclaredEnum(enumIndex);
				m_enums[enumIndex] = new Enum;
				Enum& edst = *m_enums[enumIndex];
				edst.m_name = intern(esrc.getName());
				edst.m_items._setSize(allocator, esrc.getNumItems());
				for( int itemIndex = 0; itemIndex < esrc.getNumItems(); ++itemIndex )
				{
					const hkClassEnum::Item& isrc = esrc.getItem(itemIndex);
					Enum::Item& idst = edst.m_items[itemIndex];
					idst.name = intern(isrc.getName());
					idst.value = isrc.getValue();
				}
			}


			hkTypeManager& typeManager = m_world->m_typeManager;

			m_memberInfo._reserve(allocator, klass.getNumDeclaredMembers());
			for( int memberIndex = 0; memberIndex < klass.getNumDeclaredMembers(); ++memberIndex )
			{
				const hkClassMember& msrc = klass.getDeclaredMember(memberIndex);
				if( msrc.getFlags().get(hkClassMember::SERIALIZE_IGNORED) == 0 )
				{
					MemberInfo& mdst = m_memberInfo._expandOne(allocator);
					mdst.m_name = intern(msrc.getName());

					
					hkDataObject::Type dtype = hkDataObjectUtil::getTypeFromMemberType(typeManager, msrc.getType(), msrc.getSubType(), msrc.getClass(), msrc.getCstyleArraySize() );
					HK_ASSERT(0x6fbd8ceb, !dtype->isVoid());
					
					// Replace the terminal type if attribute indicates needing a different type name
					if( const hkVariant* typeAttr = msrc.getAttribute(s_hkDataObjectTypeAttributeID) )
					{
						const hkClassMemberAccessor attrTypeName(*typeAttr, "typeName");
						const char* clsName = attrTypeName.asCstring();
						
						hkTypeManager::Type* newTerm = typeManager.addClass(clsName);
						dtype = typeManager.replaceTerminal(dtype, newTerm);
					}
					
					mdst.m_type = dtype;
					hkTypedUnion un;
					klass.getDeclaredDefault(memberIndex, un);
					mdst.m_default.i = 0; //todo
				}
			}
		}

		~hkDataClassDict()
		{
			selfDestruct();
		}

		virtual hkDataWorldDict* getWorld() const
		{
			return m_world;
		}

		virtual const char* getName() const
		{
			return m_name.getString();
		}

		virtual int getVersion() const
		{
			return m_version;
		}

		virtual hkDataClassImpl* getParent() const
		{
			return m_parent;
		}

		virtual hkBool isSuperClass(const hkDataClassImpl* k) const
		{
			const hkDataClassImpl* c = k;
			while( c )
			{
				InternedString internedClassName = intern(c->getName());
				if( internedClassName.getString() == m_name.getString() )
				{
					return true;
				}
				c = c->getParent();
			}
			return false;
		}

		// this class only
		virtual int getNumDeclaredMembers() const
		{
			return m_memberInfo.getSize();
		}
		virtual int getDeclaredMemberIndexByName(const char* nameIn) const
		{
			InternedString internedMemberName = intern(nameIn);
			for( int i = 0; i < m_memberInfo.getSize(); ++i )
			{
				if( m_memberInfo[i].m_name == internedMemberName )
				{
					return i;
				}
			}
			return -1;
		}
		virtual void getDeclaredMemberInfo(int i, hkDataClass::MemberInfo& info) const
		{
			const MemberInfo& mi = m_memberInfo[i];
			info.m_name = mi.m_name.getString();
			info.m_owner = this;
			info.m_type = mi.m_type;
			info.m_valuePtr = mi.m_valuePtr;
		}

		// all members
		virtual int getNumMembers() const
		{
			const hkDataClassDict* c = this->m_parent;
			int ret = m_memberInfo.getSize();
			while( c )
			{
				ret += c->m_memberInfo.getSize();
				c = c->m_parent;
			}
			return ret;
		}

		virtual int getMemberIndexByName(const char* nameIn) const
		{
			InternedString internedMemberName = intern(nameIn);
			const hkDataClassDict* c = this;
			while( c )
			{
				for( int i = 0; i < c->m_memberInfo.getSize(); ++i )
				{
					if( c->m_memberInfo[i].m_name == internedMemberName )
					{
						return c->getNumMembers() - c->getNumDeclaredMembers() + i;
					}
				}
				c = c->m_parent;
			}
			return -1;
		}

		virtual void getMemberInfo(int memberIndex, hkDataClass::MemberInfo& info) const
		{
			int numMembers = getNumMembers(); // TODO coalesce
			HK_ASSERT(0x275d8b19, memberIndex >= 0 && memberIndex < numMembers );
			const hkDataClassDict* c = this;
			int localIndex = memberIndex - numMembers;
			while( c )
			{
				localIndex += c->m_memberInfo.getSize();
				if( localIndex >= 0 )
				{
					const MemberInfo& mi = c->m_memberInfo[localIndex];
					mi.getMemberInfo(c, info);
					return;
				}
				c = c->m_parent;
			}
			HK_ASSERT2(0x1036239f, 0, "notreached");
		}

		virtual void getAllMemberInfo(hkArrayBase<hkDataClass::MemberInfo>& infos) const
		{
			HK_ASSERT(0xabc0011, infos.getSize() == getNumMembers());

			hkDataClass::MemberInfo* curInfos = infos.end();
			
			const hkDataClassDict* cur = this;
			while (cur)
			{
				const hkArrayBase<hkDataClassDict::MemberInfo>& mems = cur->m_memberInfo;
				const int numMem = mems.getSize();

				const hkDataClassDict::MemberInfo* curMem = mems.begin();

				curInfos -= numMem;
				for (int i = 0; i < numMem; i++)
				{
					curMem[i].getMemberInfo(this, curInfos[i]);
				}

				// next
				cur = cur->m_parent;
			}
			HK_ASSERT(0x24324a32, infos.begin() == curInfos);
		}

		void _getAllMemberHandles(hkArrayBase<hkDataObject::MemberHandle>& handles) const
		{
			HK_ASSERT(0xabcd011, handles.getSize() == getNumMembers());

			hkDataObject::MemberHandle* curHandle = handles.end();

			const hkDataClassDict* cur = this;
			while (cur)
			{
				const hkArrayBase<hkDataClassDict::MemberInfo>& mems = cur->m_memberInfo;
				const int numMem = mems.getSize();

				const hkDataClassDict::MemberInfo* curMem = mems.begin();

				curHandle -= numMem;
				for (int i = 0; i < numMem; i++, curMem++)
				{
					curHandle[i] = (hkDataObject::MemberHandle)curMem;
				}
			
				// next
				cur = cur->m_parent;
			}
			HK_ASSERT(0x24324a32, handles.begin() == curHandle);
		}

		void setVersion(int v)
		{
			m_version = v;
		}

 		static int _findDeclaredMemberByName(InternedString name, hkArrayBase<MemberInfo>& info )
 		{
 			for( int i = info.getSize() - 1; i >= 0; --i )
 			{
 				if( info[i].m_name == name  )
 				{
 					return i;
 				}
 			}
 			return -1;
 		}


		void addMember(InternedString name, hkDataObject::Type type, const void* valuePtr)
		{
			MemberInfo m;
			m.m_name = name;
			m.m_type = type;
			m.m_default.p = HK_NULL;
			m.m_valuePtr = valuePtr;
			HK_ASSERT2(0x2d59ca84, _findDeclaredMemberByName(m.m_name, m_memberInfo) == -1, "already have member" );
			m_memberInfo._pushBack(getAllocator(), m);
		}

		hkResult removeMember(InternedString name)
		{
			int memIndex = _findDeclaredMemberByName(name, m_memberInfo);
			HK_ASSERT2(0x2d59ca84, memIndex != -1, "dont have member to remove" );
			m_memberInfo.removeAtAndCopy(memIndex);
			return getMemberInfoByName(name) == HK_NULL ? HK_SUCCESS : HK_FAILURE;
		}

		void showMember(InternedString name)
 		{
 			int idx = _findDeclaredMemberByName(name, m_hiddenMemberInfo);
 			HK_ASSERT(0x57417ae2, idx >= 0);
			m_memberInfo._expandOne(getAllocator()) = m_hiddenMemberInfo[idx];
			m_hiddenMemberInfo.removeAtAndCopy(idx);
 		}

		void hideMember(InternedString name)
		{
			int idx = _findDeclaredMemberByName(name, m_memberInfo);
			HK_ASSERT(0x57417ae1, idx >= 0);
			m_hiddenMemberInfo._expandOne(getAllocator()) = m_memberInfo[idx];
			m_memberInfo.removeAtAndCopy(idx);
		}

		void renameMember(InternedString oldName, InternedString newName)
		{
			int idx = _findDeclaredMemberByName(oldName, m_memberInfo);
			HK_ASSERT(0x57417ae0, idx >= 0);
			m_memberInfo[idx].m_name = newName;
		}

		void setMemberDefault(InternedString name, const void* valuePtr)
		{
			int idx = _findDeclaredMemberByName(name, m_memberInfo);
			HK_ASSERT(0x497040d1, idx >= 0);
			m_memberInfo[idx].m_valuePtr = valuePtr;
		}

		const MemberInfo& getMemberInfoByIndex(int memberIndex) const
		{
			int numMembers = getNumMembers();
			HK_ASSERT(0x275d8b19, memberIndex >= 0 && memberIndex < numMembers );
			int localIndex = memberIndex - numMembers;
			const hkDataClassDict* c = this;
			while( c )
			{
				localIndex += c->m_memberInfo.getSize();
				if( localIndex >= 0 )
				{
					return c->m_memberInfo[localIndex];
				}
				c = c->m_parent;
			}
			HK_ASSERT2(0x1036239f, 0, "notreached");
			return m_memberInfo[-1];
		}

		const MemberInfo* getMemberInfoByName(InternedString name) const
 		{
			const hkDataClassDict* cur = this;
 			while( cur )
 			{
				for( int i = 0; i < cur->m_memberInfo.getSize(); ++i )
				{
					if( cur->m_memberInfo[i].m_name == name )
					{
						return cur->m_memberInfo.begin() + i;
					}
				}
 				cur = cur->m_parent;
 			}
 			return HK_NULL;
 		}

 		void setParent( hkDataClassDict* newParent )
 		{
			m_parent = newParent;
 		}

		void selfDestruct()
		{
			hkMemoryAllocator& mall = getAllocator();
			m_memberInfo._clearAndDeallocate(mall);
			for( int enumIndex = 0; enumIndex < m_enums.getSize(); ++enumIndex )
			{
				Enum* e = m_enums[enumIndex];
				e->m_items._clearAndDeallocate(mall);
				delete e;
			}
			m_enums._clearAndDeallocate(mall);
			m_name.clear();
		}
};

class hkDataObjectDict : public hkDataObjectImpl
{
	public:

		inline hkMemoryAllocator& getAllocator()
		{
			return *m_class->m_world->m_allocator;
		}

		HK_FORCE_INLINE hkDataWorldDict* getInnerWorld() const
		{
			return m_class->m_world;
		}

		struct MemberData
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, MemberData);
			InternedString m_memberName;
			Union m_value;
		};
		typedef hkDataClassDict::MemberInfo MemberInfo;

		hkRefPtr<const hkDataClassDict> m_class;
		hkArrayBase<MemberData> m_memberData;
		// When a member is missing from the memberData structure, it might be because
		// the member was omitted from the file this object was built from (because it's
		// value was zero), or because the object was created during patching. In this
		// case the value for the member is not known to be zero, and if never touched
		// we need to write the default value for it.
		hkBool32 m_createdDuringPatching;

		struct MemberAccess
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_SERIALIZE, MemberAccess);
			MemberAccess(const MemberInfo* m, int i) : info(m), index(i) {}
			const MemberInfo* info;
			int index;
		};


		hkDataObjectDict(const hkDataClassDict* k, bool createdDuringPatchingFlag) :
			m_class(k), m_createdDuringPatching(createdDuringPatchingFlag)
		{
		}

		~hkDataObjectDict()
		{
			selfDestruct();
		}

		virtual hkBool32 createdDuringPatching() const
		{
			return m_createdDuringPatching;
		}

		virtual hkDataObject::Handle getHandle() const
		{
			hkDataObject::Handle h;
			h.p0 = const_cast<hkDataObjectDict*>(this);
			h.p1 = 0;
			return h;
		}

		virtual const hkDataClassImpl* getClass() const
		{
			return m_class;
		}

		virtual void getAllMemberHandles(hkArrayBase<hkDataObject::MemberHandle>& handles) const
		{
			m_class->_getAllMemberHandles(handles);
		}

		void selfDestruct()
		{
			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				clear( i );
			}
			m_memberData._clearAndDeallocate(getAllocator());
		}

		virtual void destroy() 
		{ 
			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				clear( i, true );
			}
			m_memberData._clearAndDeallocate(getAllocator());
		}

		void removeMember(InternedString name)
		{
			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				if( m_memberData[i].m_memberName == name )
				{
					clear(i);
					m_memberData.removeAt(i);
					return;
				}
			}
		}

		void renameMember(InternedString oldname, InternedString newname)
		{
			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				if( m_memberData[i].m_memberName == oldname )
				{
					m_memberData[i].m_memberName = newname;
					return;
				}
			}
		}

		hkDataObject::Iterator getMemberIterator() const
		{
			return 0;
		}
		hkBool32 isValid(Iterator it) const
		{
			return it < m_memberData.getSize();
		}
		Iterator getNextMember(Iterator it) const
		{
			return it + 1;
		}

		const char* getMemberName(Iterator it) const
		{
			return m_memberData[it].m_memberName.getString();
		}
		const hkDataObject::Value getMemberValue(Iterator it) const
		{
			return hkDataObject::Value(const_cast<hkDataObjectDict*>(this), (MemberHandle)m_memberData[it].m_memberName.getString() );
		}

		void clear( int oindex, hkBool32 destroyStructs = false )
		{
			if( const MemberInfo* minfo = m_class->getMemberInfoByName( m_memberData[oindex].m_memberName ) )
			{
				Union& value = m_memberData[oindex].m_value;
				value.clear( minfo->m_type, destroyStructs );
			}
		}

		InternedString intern(const char* name) const
		{
			return m_class->intern(name);
		}

		HK_NEVER_INLINE MemberAccess _readAccess( MemberHandle handle, hkTypeManager::SubType subType )
		{
			// does it exist in the class?
			const MemberInfo* minfo = (const MemberInfo*)handle;
			HK_ASSERT(0x2432432, minfo);

			HK_ASSERT2(0x5310a16d, typesCompatible(minfo->m_type, subType), "Type mismatch" );

			int index = -1;
			// do we already have a value?
			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				//TODO typecheck, check not stale
				if( m_memberData[i].m_memberName == minfo->m_name )
				{
					index = i;
				}
			}
			if( index == -1 ) // nope, create new
			{
				index = m_memberData.getSize();
				MemberData& md = m_memberData._expandOne(getAllocator());
				md.m_memberName = minfo->m_name;

				hkDataObject::Type memType = minfo->m_type;
				if( memType->isArray())
				{
					hkDataObject::Type itemType = memType->getParent();
					md.m_value.a = hkDataArrayDict_create(getInnerWorld(), itemType, 0 );
					md.m_value.a->addReference();
				}
				else if( memType->isTuple())
				{
					hkDataObject::Type itemType = memType->getParent();
					md.m_value.a = hkDataArrayDict_create(getInnerWorld(), itemType, 0 );
					md.m_value.a->setSize( memType->getTupleSize() );
					md.m_value.a->addReference();
					
					// If default, copy over the values
					if(minfo->m_valuePtr && itemType->getSubType() == hkTypeManager::SUB_TYPE_REAL)
					{
						const int size = memType->getTupleSize();
						const hkReal* src = (const hkReal*)minfo->m_valuePtr;
						for (int i = 0; i < size; i++)
						{
							md.m_value.a->setReal(i, src[i]);
						}
					}
				}
				else if( memType->isPointer())
				{
					md.m_value.o = HK_NULL;
				}
				else if( memType->isClass() )
				{
					// Create an empty dataobject in case someone wants to access it
					// before it is finalized (in a following patch)
					hkDataClass dataClass(m_class->m_world->findClass(memType->getTypeName()));
					hkDataObject impl(m_class->m_world->newObject(dataClass, true));
					md.m_value.o = impl.getImplementation();
					md.m_value.o->addReference();
				}
				else
				{
					// If we have a value from a patch, use it
					if(minfo->m_valuePtr)
					{
						hkTypeManager::SubType memSubType = memType->getSubType();

						if(memSubType == hkTypeManager::SUB_TYPE_BYTE || memSubType == hkTypeManager::SUB_TYPE_INT)
						{
							md.m_value.i = *(reinterpret_cast<const hkInt64*>(minfo->m_valuePtr));
						}
						else if(memSubType == hkTypeManager::SUB_TYPE_REAL)
						{
							md.m_value.r = *(reinterpret_cast<const hkReal*>(minfo->m_valuePtr));
						}
						else if(memSubType == hkTypeManager::SUB_TYPE_CSTRING)
						{
							md.m_value.s = hkString::strDup(reinterpret_cast<const char*>(minfo->m_valuePtr));
						}
						else
						{
							HK_ASSERT2(0x5cef873c, 0, "Unsupported type for patch value member assignment");
						}
					}
					else
					{
						md.m_value.i = 0;
					}
				}
			}
			return MemberAccess(minfo, index);
		}

		MemberAccess _writeAccess( MemberHandle handle, hkTypeManager::SubType subType)
		{
			const MemberInfo* minfo = (const MemberInfo*)handle;
			HK_ASSERT(0x24324324, minfo);

			HK_ASSERT2(0x3a26aa87, typesCompatible(minfo->m_type, subType), "Type mismatch");
			for( int i = 0; i < m_memberData.getSize(); ++i ) // do we already have a value?
			{
				if( m_memberData[i].m_memberName == minfo->m_name )
				{
					m_memberData[i].m_value.clear( minfo->m_type );
					return MemberAccess(minfo, i);
				}
			}
			// no, create new blank slot
			int index = m_memberData.getSize();
			MemberData& md = m_memberData._expandOne(getAllocator());
			md.m_memberName = minfo->m_name;
			md.m_value.i = 0;
			return MemberAccess(minfo, index);
		}

		hkBool32 _hasMember(InternedString name) const
		{
			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				if( m_memberData[i].m_memberName == name )
				{
					return true;
				}
			}
			// If the member has an unrealized default, it is still a member
			// as an attempt to read it will succeed
			if( const MemberInfo* minfo = m_class->getMemberInfoByName( name ) )
			{
				return (minfo->m_valuePtr != HK_NULL);
			}
			return false;
		}
		hkBool32 hasMember(const char* name) const
		{
			return _hasMember( intern(name) );
		}

		hkBool32 isSet(MemberHandle handle)
		{
			const MemberInfo* minfo = (const MemberInfo*)handle;

			// If the member has an unrealized default, it is still a member
			// as an attempt to read it will succeed
			if (minfo->m_valuePtr != HK_NULL)
			{
				return true;
			}

			for( int i = 0; i < m_memberData.getSize(); ++i )
			{
				if( m_memberData[i].m_memberName == minfo->m_name )
				{
					return true;
				}
			}
			return false;
		}

		hkDataObject::Value accessByName(const char* name)
		{
			InternedString internedName = intern(name);
			const hkDataClassDict::MemberInfo* info = m_class->getMemberInfoByName(internedName);
			if (info)
			{
				HK_ASSERT(0x3423432a, internedName.getString() == info->m_name.getString());
				return hkDataObject::Value(this, (MemberHandle)info);
			}
			else
			{
				return hkDataObject::Value(HK_NULL, HK_NULL);
			}
		}

		// getters

		hkDataArrayImpl* asArray( MemberHandle handle)
		{
			const MemberAccess mem = _readAccess(handle, hkTypeManager::SUB_TYPE_ARRAY);
			const Union& value = m_memberData[mem.index].m_value;
			return value.a;
		}

		const char* asString( MemberHandle handle )
		{
			const MemberAccess mem = _readAccess(handle, hkTypeManager::SUB_TYPE_CSTRING);
			const Union& value = m_memberData[mem.index].m_value;
			return value.s;
		}

		hkInt64 asInt( MemberHandle handle )
		{
			const MemberAccess mem = _readAccess(handle, hkTypeManager::SUB_TYPE_VOID);
			const Union& value = m_memberData[mem.index].m_value;
			switch(mem.info->m_type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_BYTE:
				case hkTypeManager::SUB_TYPE_INT: return value.i;
				default: HK_ASSERT(0x24e845e5, 0); return 0;
			}
		}

		hkDataObjectImpl* asObject( MemberHandle handle )
		{
			MemberAccess mem = _readAccess(handle, hkTypeManager::SUB_TYPE_CLASS);
			return m_memberData[mem.index].m_value.o;
		}

		const hkReal* asVec( MemberHandle handle, int nreal )
		{
			MemberAccess mem = _readAccess(handle, hkTypeManager::SUB_TYPE_TUPLE);

			const Union& value = m_memberData[mem.index].m_value;

			return hkRealArrayImplementation_getReals(value.a, nreal);
		}

		hkReal asReal( MemberHandle handle )
		{
			MemberAccess mem = _readAccess(handle, hkTypeManager::SUB_TYPE_REAL);
			return m_memberData[mem.index].m_value.r;
		}

		// assignments
		void assign(  MemberHandle handle, const hkDataObject::Value& value )
		{
			hkDataObjectDict* dict = static_cast<hkDataObjectDict*>(value.m_impl);
			MemberAccess src = dict->_readAccess(value.m_handle, hkTypeManager::SUB_TYPE_VOID);
			MemberAccess dst = _writeAccess(handle, hkTypeManager::SUB_TYPE_VOID);

			const Union val = dict->m_memberData[src.index].m_value;

			// Check if the assignment is possible
			hkDataObject::Type srcType = src.info->m_type;
			hkDataObject::Type dstType = dst.info->m_type;

			

			if (srcType->isArray() || srcType->isTuple())
			{
				//&& srcType->getParent() == dstType->getParent())

				if (dstType->isArray() || dstType->isTuple())
				{
					if (val.a)
					{
						// Check it is the right type
						val.a->addReference();
					}
				}
				else
				{
					HK_ASSERT(0x234324, !"Can't set type");
				}
			}
			// && dstType->getParent() == srcType->getParent()
			else if ((srcType->isPointer() && dstType->isPointer()) ||
					 (srcType->isClass() && srcType == dstType))
			{
				if (val.o)
				{
					val.o->addReference();
				}
			}

			m_memberData[dst.index].m_value = dict->m_memberData[src.index].m_value;
		}

		void assign( MemberHandle handle, hkDataObjectImpl* value )
		{
			if( value )
			{
				value->addReference();
			}
			MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_CLASS);
			m_memberData[mem.index].m_value.o = value;
		}

		void assign( MemberHandle handle, hkDataArrayImpl* value )
		{
			HK_ASSERT(0x7cbaa60b, value);
			value->addReference();
			MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_VOID);

			hkDataObject::Type valueType = value->getType();
			hkDataObject::Type dstType = mem.info->m_type->getParent();

			if (dstType->isEqual(valueType))
			{
				m_memberData[mem.index].m_value.a = value;
			}
			else
			{
				if (dstType->isClass() && dstType->getTypeName() == HK_NULL && valueType->isClass())
				{
					m_memberData[mem.index].m_value.a = value;
				}
				else
				{
					HK_ASSERT(0x3243243, !"Array assignment of different types");
				}
			}
		}

		void assign( MemberHandle handle, const char* value )
		{
			MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_CSTRING);
			m_memberData[mem.index].m_value.s = value ? hkString::strDup(value) : HK_NULL;
		}

		void assign( MemberHandle handle, hkReal r )
		{
			MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_REAL);
			m_memberData[mem.index].m_value.r = r;
		}

		void assign( MemberHandle handle, hkHalf r )
		{
			MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_REAL);
			m_memberData[mem.index].m_value.r = r;
		}

		void assign( MemberHandle handle, const hkReal* r, int nreal )
		{
			MemberAccess mem = _writeAccess( handle, hkTypeManager::SUB_TYPE_VOID);

			hkDataObject::Type type = mem.info->m_type;
			if (type->isTuple() && type->getParent()->isReal())
			{
				Union& u = m_memberData[mem.index].m_value;
				// I have to create the array which will store the tuples
				if (u.a == HK_NULL)
				{
					u.a = hkDataArrayDict_create(m_class->m_world, type->getParent());
					u.a->setSize(type->getTupleSize());
					u.a->addReference();
				}

				hkReal* dst = hkRealArrayImplementation_getReals(u.a, nreal);
				HK_ASSERT(0x24324324, dst);
				hkString::memCpy(dst, r, sizeof(hkReal) * nreal);
			}
			else
			{
				HK_ASSERT(0x213131a, !"Cannot assign vec to this type");
			}
		}

		void assign( MemberHandle handle, hkInt64 valueIn )
		{
			const MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_VOID);
			HK_ASSERT(0x75389b1a, mem.info && mem.index != -1);
			MemberData& mdata = m_memberData[mem.index];
			switch(mem.info->m_type->getSubType())
			{
				case hkTypeManager::SUB_TYPE_BYTE:
				case hkTypeManager::SUB_TYPE_INT:
					mdata.m_value.i = valueIn;
					break;
				case hkTypeManager::SUB_TYPE_REAL:
					mdata.m_value.r = hkReal(valueIn);
					break;
				default:
					HK_ASSERT2(0x23b750e2, 0, "Mismatched types");//m_class->m_memberInfo[cindex].m_type == TYPE_INT );
			}
		}
		
		void assign( MemberHandle handle, int valueIn )
		{
			const MemberAccess mem = _writeAccess(handle, hkTypeManager::SUB_TYPE_VOID);
			HK_ASSERT(0x75389b1a, mem.info && mem.index != -1);
			MemberData& mdata = m_memberData[mem.index];
			switch(mem.info->m_type->getSubType())
			{
			case hkTypeManager::SUB_TYPE_BYTE:
			case hkTypeManager::SUB_TYPE_INT:
				mdata.m_value.i = valueIn;
				break;
			case hkTypeManager::SUB_TYPE_REAL:
				mdata.m_value.r = hkReal(valueIn);
				break;
			default:
				HK_ASSERT2(0x23b750e2, 0, "Mismatched types");//m_class->m_memberInfo[cindex].m_type == TYPE_INT );
			}
		}
		void getMemberInfo(MemberHandle handle, hkDataClass::MemberInfo& infoOut)
		{
			const MemberInfo* minfo = (const MemberInfo*)handle;
			minfo->getMemberInfo(HK_NULL, infoOut);
		}
};


template< typename HKTYPE>
struct BasicArrayImplementation : public hkDataArrayImpl
{
	hkArrayBase<HKTYPE> m_data;
	hkDataWorldDict* m_world;
	hkDataObject::Type m_type;

	BasicArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : m_world(world), m_type(type) {}
	~BasicArrayImplementation() { m_data._clearAndDeallocate(getAllocator()); }

	hkMemoryAllocator& getAllocator() const
	{
		return *m_world->m_allocator;
	}
	virtual hkDataObject::Type getType() const
	{
		return m_type;
	}
	virtual hkDataWorld* getWorld() const 
	{
		return m_world;
	}
	virtual const hkDataClassImpl* getClass() const
	{
		return HK_NULL;
	}
	virtual void clear()
	{
		m_data.clear();
	}
	virtual void reserve( int n )
	{
		m_data._reserve(getAllocator(), n);
	}
	virtual void setSize( int n )
	{
		m_data._setSize(getAllocator(), n, 0);
	}
	virtual int getSize() const
	{
		return m_data.getSize();
	}
};

struct VariableIntArrayImplementation : public hkDataArrayImpl
{
	enum {
		ARRAY_TYPE_INT32,
		ARRAY_TYPE_INT64
	};

	VariableIntArrayImplementation(hkDataWorldDict* world) : 
		m_world(world),
		m_arrayType(ARRAY_TYPE_INT32), 
		m_assigned(false)
	{}
	~VariableIntArrayImplementation()
	{
		m_int32Array._clearAndDeallocate(_getAllocator());
		m_int64Array._clearAndDeallocate(_getAllocator());
	}

	// hkDataArrayImpl
	virtual hkDataObject::Type getType() const
	{
		return m_world->m_typeManager.getSubType(hkTypeManager::SUB_TYPE_INT);
	}

	virtual int getUnderlyingIntegerSize() const
	{
		if(m_arrayType == ARRAY_TYPE_INT32)
		{
			return hkSizeOf(hkInt32);
		}
		else
		{
			return hkSizeOf(hkInt64);
		}
	}

	virtual void clear()
	{
		m_int32Array.clear();
		// On clearing the array will revert to 32-bit by default so release any 64-bit storage
		m_int64Array._clearAndDeallocate(_getAllocator());
		m_arrayType = ARRAY_TYPE_INT32;
		m_assigned = false;
	}

	hkDataWorld* getWorld() const
	{
		return m_world;
	}

	virtual void reserve( int n )
	{
		switch(m_arrayType)
		{
		case ARRAY_TYPE_INT32:
			m_int32Array._reserve(_getAllocator(), n);
			break;
		case ARRAY_TYPE_INT64:
			m_int64Array._reserve(_getAllocator(), n);
			break;
		default:
			HK_ASSERT2(0x5a253e88, 0, "Unknown variable array type");
			break;
		}
	}
	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) 
	{
		if (m_arrayType == ARRAY_TYPE_INT32)
		{
			arrOut.m_type = hkClassMember::TYPE_INT32;
			arrOut.m_size = m_int32Array.getSize();
			arrOut.m_data = m_int32Array.begin();
			arrOut.m_stride = (int)sizeof(hkUint32);
		}
		else
		{
			arrOut.m_type = hkClassMember::TYPE_INT64;
			arrOut.m_size = m_int64Array.getSize();
			arrOut.m_data = m_int64Array.begin();
			arrOut.m_stride = (int)sizeof(hkUint64);
		}
		arrOut.m_tupleSize = 1;
		return HK_SUCCESS;
	}

	virtual void setSize(int size)
	{
		switch(m_arrayType)
		{
		case ARRAY_TYPE_INT32:
			m_int32Array._setSize(_getAllocator(), size, 0);
			break;
		case ARRAY_TYPE_INT64:
			m_int64Array._setSize(_getAllocator(), size, 0);
			break;
		default:
			HK_ASSERT2(0x7eaa9483, 0, "Unknown variable array type");
			break;
		}
		if(size == 0)
		{
			m_assigned = false;
		}
	}

	virtual int getSize() const
	{
		switch(m_arrayType)
		{
		case ARRAY_TYPE_INT32:
			return m_int32Array.getSize();
		case ARRAY_TYPE_INT64:
			return m_int64Array.getSize();
		default:
			HK_ASSERT2(0x32842288, 0, "Unknown variable array type");
			return 0;
		}
	}

	virtual int asInt(int index) const
	{
		switch(m_arrayType)
		{
		case ARRAY_TYPE_INT32:
			return m_int32Array[index];
		case ARRAY_TYPE_INT64:
			//HK_WARN_ALWAYS(0x4f908fc0, "Reading a 64-bit DataObjectArray as a 32-bit value");
			return int(m_int64Array[index]);
		default:
			HK_ASSERT2(0x163e6a8d, 0, "Unknown variable array type");
			return 0;
			break;
		}
	}

	virtual hkInt64 asInt64(int index) const
	{
		switch(m_arrayType)
		{
		case ARRAY_TYPE_INT32:
			//HK_WARN_ALWAYS(0x193b8e2a, "Reading a 32-bit DataObjectArray as a 64-bit value");
			return m_int32Array[index];
		case ARRAY_TYPE_INT64:
			return m_int64Array[index];
		default:
			HK_ASSERT2(0x75f2ec68, 0, "Unknown variable array type");
			return 0;
		}
	}

	virtual void setHalf(int index, hkHalf val)
	{
		HK_ASSERT(0x75f2ec69, m_arrayType==ARRAY_TYPE_INT32);
		#if defined(HK_HALF_IS_FLOAT)
			HK_ASSERT(0x75f2ec70, 0);
		#endif
		hkUint16 u = *reinterpret_cast<hkUint16*>(&val);
		m_int32Array[index] = u;
	}

	virtual hkHalf asHalf(int index) const
	{
		HK_ASSERT(0x75f2ec69, m_arrayType==ARRAY_TYPE_INT32);
		#if defined(HK_HALF_IS_FLOAT)
			HK_ASSERT(0x75f2ec70, 0);
		#endif
		hkUint16 u = hkUint16( m_int32Array[index] );
		return *reinterpret_cast<hkHalf*>(&u);
	}

	virtual void setInt(int index, int val)
	{
		if(m_arrayType == ARRAY_TYPE_INT32)
		{
			m_int32Array[index] = val;
		}
		else
		{
			setInt64(index, hkInt64(val));
		}
		m_assigned = true;
	}

	void _reallocateArray()
	{
		HK_ASSERT2(0x1ca62e23, m_arrayType == ARRAY_TYPE_INT32, "Reallocating array that is not 32-bit");
		if(m_assigned)
		{
			HK_WARN_ALWAYS(0x3599e1ed, "Converting DataObjectArray from 32-bit to 64-bit. If this is intended, always use 64-bit values");
		}

		m_int64Array._setSize(_getAllocator(), m_int32Array.getSize());
		for(int i=0;i<m_int32Array.getSize();i++)
		{
			m_int64Array[i] = m_int32Array[i];
		}
		m_int32Array._clearAndDeallocate(_getAllocator());
		m_arrayType = ARRAY_TYPE_INT64;
	}

	virtual void setInt64(int index, hkInt64 val)
	{
		if(m_arrayType != ARRAY_TYPE_INT64)
		{
			_reallocateArray();
		}
		m_assigned = true;
		m_int64Array[index] = val;
	}

	using hkDataArrayImpl::setAll;

	virtual void setAll(const hkInt64* in, int size)
	{
		setSize(size);
		for(int i=0; i<size; i++)
		{
			setInt64(i, in[i]);
		}
	}

	virtual void setAll(const hkUint64* in, int size)
	{
		setAll(reinterpret_cast<const hkInt64*>(in), size);
	}

	virtual void setAll(const hkInt32* in, int size)
	{
		if (m_arrayType != ARRAY_TYPE_INT32)
		{
			m_arrayType = ARRAY_TYPE_INT32;
			m_int64Array._clearAndDeallocate(_getAllocator());	
		}
		m_int32Array._setSize(_getAllocator(), size, 0);
		hkString::memCpy(m_int32Array.begin(), in, sizeof(hkInt32) * size);
	}

	virtual void setAll(const hkUint32* in, int size)
	{
		setAll(reinterpret_cast<const hkInt32*>(in), size);
	}

	virtual void setAll(const hkInt16* in, int size)
	{
		if (m_arrayType != ARRAY_TYPE_INT32)
		{
			m_arrayType = ARRAY_TYPE_INT32;
			m_int64Array._clearAndDeallocate(_getAllocator());	
		}
		m_int32Array._setSize(_getAllocator(), size, 0);
		for(int i=0; i<size; i++)
		{
			setInt(i, in[i]);
		}
	}

	virtual void setAll(const hkUint16* in, int size)
	{
		if (m_arrayType != ARRAY_TYPE_INT32)
		{
			m_arrayType = ARRAY_TYPE_INT32;
			m_int64Array._clearAndDeallocate(_getAllocator());	
		}
		m_int32Array._setSize(_getAllocator(), size, 0);
		for(int i=0; i<size; i++)
		{
			// Calling setInt will NOT cause sign bit extension because 
			// in[i] is an unsigned 16 bits integer. We have to avoid sign
			// extension because this will alter the value in memory when
			// interpreted as unsigned.
			setInt(i, in[i]);
		}
	}

	virtual const hkDataClassImpl* getClass() const
	{
		return HK_NULL;
	}

	virtual int getIntType()
	{
		return m_arrayType;
	}

protected:

	hkDataWorldDict* m_world;
	hkMemoryAllocator& _getAllocator() const { return *m_world->m_allocator; }
	int m_arrayType;
	hkArrayBase<hkInt32> m_int32Array;
	hkArrayBase<hkInt64> m_int64Array;
	//hkMemoryAllocator* m_allocator;
	hkBool m_assigned;
};


struct ByteArrayImplementation : public BasicArrayImplementation<hkUint8>
{
	ByteArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : BasicArrayImplementation<hkUint8>(world, type) {}
	virtual int asInt(int i) const { return m_data[i]; }
	virtual void setInt(int i, int val) { m_data[i] = hkUint8(val); }
	virtual void setInt64(int i, hkInt64 val) { m_data[i] = hkUint8(val); }

	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) 
	{
		arrOut.m_type = hkClassMember::TYPE_UINT8;
		arrOut.m_size = m_data.getSize();
		arrOut.m_data = m_data.begin();
		arrOut.m_tupleSize = 1;
		arrOut.m_stride = (int)sizeof(hkUint8);
		return HK_SUCCESS;
	}
	using hkDataArrayImpl::setAll;

	virtual void setAll(const hkUint8* vals, int n)
	{
		m_data._setSize(getAllocator(), n);
		hkString::memCpy(m_data.begin(), vals, n * sizeof(hkUint8));
	}
};

struct ArrayOfTuplesImplementation : public hkDataArrayImpl
{
	struct View : public hkDataArrayImpl
	{
		View(ArrayOfTuplesImplementation* owner, int i):
			m_owner(owner),
			m_baseIndex(i)
		{
			HK_ASSERT(0x32432432, owner->getType()->isTuple());
		}
		
		virtual hkDataArrayImpl* swizzleObjectMember(const char* name) const
		{
			ArrayOfTuplesImplementation* impl = (ArrayOfTuplesImplementation*)( m_owner->swizzleObjectMember(name) );
			return new View(impl, m_baseIndex);
		}

		virtual hkDataObject::Type getType() const
		{
			return m_owner->getType()->getParent();
		}
		virtual hkDataWorld* getWorld() const
		{
			return m_owner->getWorld();
		}
		virtual hkDataClassImpl* getClass() const
		{
			hkDataObject::Type type = getType();
			if (type->isClass())
			{
				// Look up the class
				hkDataClassImpl* cls = m_owner->m_world->findClass(type->getTypeName());
				if (!cls)
				{
					HK_WARN(0x2432a4aa, "Couldn't find class '" << type->getTypeName() << "'");
					return HK_NULL;
				}
				return cls;
			}
			return HK_NULL;
		}
		virtual void clear()
		{
			m_owner->m_impl->clear();
		}
		virtual void reserve( int n )
		{
			HK_ASSERT(0x692ae952, n < getSize());
		}
		virtual void setSize(int n)
		{
			HK_ASSERT(0x29fd8d43, n == getSize());
		}
		virtual int getSize() const
		{
			return m_owner->getType()->getTupleSize();
		}

		using hkDataArrayImpl::setAll;

		virtual void setAll(const hkUint8* v, int n)
		{
			HK_ASSERT(0x6f81aeef, m_owner->m_impl->getSize() >= (m_baseIndex + n));
			for( int index = 0; index < n; ++index )
			{
				m_owner->m_impl->setInt(m_baseIndex+index, v[index]);
			}
		}
		virtual void setAll(const hkReal* v, int n)
		{
			HK_ASSERT(0x6f81aeef, m_owner->m_impl->getSize() >= (m_baseIndex + n));
			for( int index = 0; index < n; ++index )
			{
				m_owner->m_impl->setReal(m_baseIndex+index, v[index]);
			}
		}
		virtual void setAll(const hkInt32* v, int n)
		{
			HK_ASSERT(0x6f81aeef, m_owner->m_impl->getSize() >= (m_baseIndex + n));
			for( int index = 0; index < n; ++index )
			{
				m_owner->m_impl->setInt(m_baseIndex+index, v[index]);
			}
		}

		hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) { return HK_FAILURE; }

#		define ITEM_ACCESSOR(TYPE, NAME, UMEMBER) \
			virtual TYPE as##NAME(int index) const { return m_owner->m_impl->as##NAME(m_baseIndex+index); } \
			virtual void set##NAME(int index, TYPE val) { m_owner->m_impl->set##NAME(m_baseIndex+index,val); }
		ITEM_ACCESSOR(const hkReal*, Vec, ra);
		ITEM_ACCESSOR(const char*, String, s);
		ITEM_ACCESSOR(hkReal, Real, r);
		ITEM_ACCESSOR(int, Int, i);
		ITEM_ACCESSOR(hkInt64, Int64, i);
		ITEM_ACCESSOR(hkDataObjectImpl*, Object, o);
		ITEM_ACCESSOR(hkDataArrayImpl*, Array, a);
#		undef ITEM_ACCESSOR

	private:
		ArrayOfTuplesImplementation* m_owner;
		int m_baseIndex;
	};

	ArrayOfTuplesImplementation( hkDataWorldDict* world, hkDataArrayImpl* arr, int tc )
		: m_world(world), m_impl(arr), m_tupleCount(tc), m_swizzle(HK_NULL)
	{
		if ( m_impl != HK_NULL )
		{
			m_impl->addReference();
		}
		HK_ASSERT(0x50fb134c, tc > 0 );
	}
	~ArrayOfTuplesImplementation()
	{
		if ( m_impl != HK_NULL )
		{
			m_impl->removeReference();
		}

		if ( m_swizzle != HK_NULL )
		{
			m_swizzle->removeReference();
		}
	}

	void setImpl( hkDataArrayImpl* impl )
	{
		if ( impl != HK_NULL )
		{
			//HK_ASSERT(0x50fb134b, !impl->getType()->isTuple());
			impl->addReference();
		}

		if ( m_impl != HK_NULL )
		{
			m_impl->removeReference();
		}

		m_impl = impl;
	}

	virtual hkDataArrayImpl* swizzleObjectMember(const char* name) const
	{
		if ( m_swizzle == HK_NULL )
		{
			m_swizzle = new ArrayOfTuplesImplementation(m_world, HK_NULL, m_tupleCount );
			m_swizzle->addReference();
		}
		m_swizzle->setImpl( m_impl->swizzleObjectMember(name) );
		return m_swizzle;
	}

	virtual hkDataObject::Type getType() const
	{
		hkDataObject::Type type = m_impl->getType();
		return m_world->getTypeManager().makeTuple(type, m_tupleCount);
	}
	virtual hkDataWorld* getWorld() const
	{
		return m_world;
	}

	virtual hkDataClassImpl* getClass() const
	{
		hkDataObject::Type type = getType();
		if (type->isClass())
		{
			hkDataClassImpl* cls = m_world->findClass(type->getTypeName());
			HK_ASSERT(0x234a3243, cls);
			return cls;
		}
		return HK_NULL;
	}
	virtual void clear()
	{
		m_impl->clear();
	}
	virtual void reserve( int n )
	{
		m_impl->reserve( n * m_tupleCount );
	}
	virtual void setSize(int n)
	{
		m_impl->setSize( n * m_tupleCount);
	}
	virtual int getSize() const
	{
		return m_impl->getSize() / m_tupleCount;
	}

	virtual hkDataArrayImpl* asArray(int idx) const
	{
		ArrayOfTuplesImplementation& self = *const_cast<ArrayOfTuplesImplementation*>(this);

		return new View(&self, idx * self.m_tupleCount);
	}

	virtual void setArray(int idx, hkDataArrayImpl* src)
	{
		ArrayOfTuplesImplementation& self = *const_cast<ArrayOfTuplesImplementation*>(this);

		View view(this, idx * self.m_tupleCount);

		switch( src->getType()->getSubType() )
		{
#define SET_ARRAY_ITEMS(ITEM_TYPE, FUNC) \
			case hkTypeManager::SUB_TYPE_##ITEM_TYPE: \
			{ \
				for( int i = 0; i < self.m_tupleCount; ++i ) \
				{ \
					view.set##FUNC( i , src->as##FUNC(i) ); \
				} \
				break; \
			}
			SET_ARRAY_ITEMS(BYTE, Int)
			//SET_ARRAY_ITEMS(TYPE_INT, Int64)
			SET_ARRAY_ITEMS(REAL, Real)
			SET_ARRAY_ITEMS(CLASS, Object)
			SET_ARRAY_ITEMS(CSTRING, String)
			SET_ARRAY_ITEMS(POINTER, Object)
#undef SET_ARRAY_ITEMS

			// Special case for int arrays
			case hkTypeManager::SUB_TYPE_INT:
			{
				VariableIntArrayImplementation* varArray = static_cast<VariableIntArrayImplementation*>(src);

				for( int i = 0; i < self.m_tupleCount; ++i )
				{
					if(varArray->getIntType() == VariableIntArrayImplementation::ARRAY_TYPE_INT32)
					{
						view.setInt(i , src->asInt(i));
					}
					else
					{
						view.setInt64(i , src->asInt64(i));
					}
				}
				break;
			}
			// We probably need a special case for tuples... but haven't really figured out how that works yet!!!
			default:
				HK_ASSERT(0x100e179a, 0);
		}
	}
	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) { return HK_FAILURE; }

	hkDataWorldDict* m_world;
	hkDataArrayImpl* m_impl;
	int m_tupleCount;

	// We keep our own internal swizzle to be used for calls to swizzleObjectMember().  This is
	// necessary because the callee is responsible for managing the memory for swizzleObjectMember().
	mutable ArrayOfTuplesImplementation* m_swizzle;
};

struct RealArrayImplementation : public BasicArrayImplementation<hkReal>
{
	RealArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : BasicArrayImplementation<hkReal>(world, type) {}
	virtual hkReal asReal(int i) const { return m_data[i]; }
	virtual void setReal(int i, hkReal val) { m_data[i] = val; }

	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) 
	{  
		arrOut.m_data = m_data.begin();
		arrOut.m_size = m_data.getSize();
		arrOut.m_stride = int(sizeof(hkReal));
		arrOut.m_type = hkClassMember::TYPE_REAL;
		arrOut.m_tupleSize = 1;
		return HK_SUCCESS;
	}

	using hkDataArrayImpl::setAll;

	virtual void setAll(const hkReal* vals, int n)
	{
		m_data._setSize(getAllocator(), n);
		hkString::memCpy(m_data.begin(), vals, n * sizeof(hkReal));
	}
};

struct RealArrayView : public hkDataArrayImpl
{
	hkDataWorldDict* m_world;
	hkReal* m_data;
	int m_size;

	RealArrayView(hkDataWorldDict* world, hkReal* data, int size) :
		m_world(world),
		m_data(data),
		m_size(size)
	{}
	virtual hkReal asReal(int i) const { HK_ASSERT(0x234324, i >= 0 && i < m_size); return m_data[i]; }
	virtual void setReal(int i, hkReal val) { HK_ASSERT(0x234324, i >= 0 && i < m_size); m_data[i] = val; }

	virtual hkDataObject::Type getType(void) const
	{
		return m_world->getTypeManager().getSubType(hkTypeManager::SUB_TYPE_REAL);
	}
	virtual hkDataWorld* getWorld() const { return m_world; }
	virtual int getSize() const { return m_size; }
	virtual const hkDataClassImpl* getClass(void) const { return HK_NULL; }
	virtual void reserve(int) { HK_ASSERT(0x5a799026, !"NotImplemented"); }
	virtual void setSize(int) { HK_ASSERT(0x4eb7c3e3, !"NotImplemented"); }
	virtual void clear(void) { HK_ASSERT(0x5bf2c6de, !"NotImplemented"); }

	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) 
	{  
		arrOut.m_data = m_data;
		arrOut.m_size = m_size;
		arrOut.m_stride = int(sizeof(hkReal));
		arrOut.m_tupleSize = 1;
		arrOut.m_type = hkClassMember::TYPE_REAL;
		return HK_SUCCESS;
	}

	using hkDataArrayImpl::setAll;

	virtual void setAll(const hkReal* vals, int n)
	{
		HK_ASSERT(0x23423a32, n <= m_size);
		hkString::memCpy(m_data, vals, n * sizeof(hkReal));
	}
};

struct VecArrayImplementation : public hkDataArrayImpl
{
 	hkArrayBase<hkReal> m_data;
	hkDataWorldDict* m_world;
	hkDataObject::Type m_type;				

	VecArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : 
		m_world(world), m_type(type) 
	{
		HK_ASSERT(0x3243a432, type->isTuple() && type->getParent()->isReal());
	}
	~VecArrayImplementation() { m_data._clearAndDeallocate(getAllocator()); }

	hkMemoryAllocator& getAllocator() const
	{
		return *m_world->m_allocator;
	}
	virtual const hkReal* asVec(int index) const  { return &m_data[index * _getNumReal()]; }
	virtual void setVec(int index, const hkReal* val)
	{
		const int numReal = _getNumReal(); 
		for( int i = 0; i < numReal; ++i )
		{
			m_data[i + index * numReal] = val[i];
		}
	}
	virtual hkDataObject::Type getType() const
	{
		return m_type;
	}
	virtual hkDataWorld* getWorld() const
	{
		return m_world;
	}
	virtual const hkDataClassImpl* getClass() const 
	{
		return HK_NULL;
	}
	virtual void clear() { m_data.clear(); }
	virtual void reserve( int n ) { m_data._reserve(getAllocator(), n * _getNumReal()); }
	virtual void setSize( int n )
	{
		m_data._setSize( getAllocator(), n * _getNumReal(), 0 );
	}
	virtual int getSize() const { return m_data.getSize() /  _getNumReal(); }

	virtual hkDataArrayImpl* asArray(int index) const 
	{ 
		// Access the contents, as a sub array
		const int numReal = _getNumReal();
		return new RealArrayView(m_world, const_cast<hkReal*>(&m_data[index * numReal]), numReal);
	} 
	virtual void setAll(const hkReal* data, int size)
	{
		int tupleSize = _getNumReal();
		m_data._setSize( getAllocator(), size * tupleSize, 0 );
		// Copy the data over
		hkString::memCpy(m_data.begin(), data, size * tupleSize * sizeof(hkReal));
	}

	// SNC complains that hkDataArrayImpl::setAll is only partially overridden otherwise
#define ARRAY_SET(A) virtual void setAll(const A*, int n) { HK_ASSERT(0x227cd5bd, 0); }
	ARRAY_SET(hkBool);
	ARRAY_SET(char);
	ARRAY_SET(hkInt8);
	ARRAY_SET(hkUint8);
	ARRAY_SET(hkInt16);
	ARRAY_SET(hkUint16);
	ARRAY_SET(hkInt32);
	ARRAY_SET(hkUint32);
	ARRAY_SET(hkInt64);
	ARRAY_SET(hkUint64);
	ARRAY_SET(hkHalf);
#		undef ARRAY_SET

	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) 
	{  
		arrOut.m_data = m_data.begin();
		arrOut.m_size = m_data.getSize() / _getNumReal();
		arrOut.m_stride = m_type->getTupleSize() * sizeof(hkReal);
		arrOut.m_type = hkClassMember::TYPE_REAL;
		arrOut.m_tupleSize = _getNumReal();
		return HK_SUCCESS;
	}

	int _getNumReal() const { return m_type->getTupleSize(); }
};

/* static */hkReal* hkRealArrayImplementation_getReals(hkDataArrayImpl* arrayIn, int numReals)
{
	RealArrayImplementation* array = static_cast<RealArrayImplementation*>(arrayIn);

	if (array->m_data.getSize() >= numReals)
	{
		return array->m_data.begin();
	}
	return HK_NULL;
}


struct PointerArrayImplementation : public BasicArrayImplementation<hkDataObjectImpl*>
{
	PointerArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : 
		BasicArrayImplementation<hkDataObjectImpl*>(world, type)
	{
		HK_ASSERT(0x32423423, type->isPointer() && type->getParent()->isClass());
	}
	~PointerArrayImplementation()
	{
		for( int i = 0; i < m_data.getSize(); ++i )
		{
			if( m_data[i] )
			{
				m_data[i]->removeReference();
			}
		}
	}
	virtual hkDataObjectImpl* asObject(int i) const
	{
		return m_data[i];
	}
	virtual void setObject(int i, hkDataObjectImpl* val)
	{
		if( val ) val->addReference();
		if( m_data[i] ) m_data[i]->removeReference();
		m_data[i] = val;
	}
	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) { return HK_FAILURE; }
};

struct CstringArrayImplementation : public BasicArrayImplementation<char*>
{
	CstringArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : BasicArrayImplementation<char*>(world, type) { }
	~CstringArrayImplementation()
	{
		for( int i = 0; i < m_data.getSize(); ++i )
		{
			hkString::strFree( m_data[i] );
		}
	}
	virtual const char* asString(int i) const { return m_data[i]; }
	virtual void setString(int i, const char* val)
	{
		hkString::strFree( m_data[i] );
		m_data[i] = val ? hkString::strDup(val) : HK_NULL;
	}
	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) { return HK_FAILURE; }
};

struct ArrayArrayImplementation : public BasicArrayImplementation<hkDataArrayImpl*>
{
	typedef BasicArrayImplementation<hkDataArrayImpl*> Parent;
	ArrayArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type) : 
		BasicArrayImplementation<hkDataArrayImpl*>(world, type)
	{
	}

	~ArrayArrayImplementation()
	{
		clear();
	}

	virtual void clear()
	{
		for (int i = 0; i < m_data.getSize(); i++)
		{
			hkDataArrayImpl* impl = m_data[i];
			if (impl)
			{
				impl->removeReference();
			}
		}
	}
	void setArray(int index, hkDataArrayImpl* impl)
	{
		if (impl)
		{
			impl->addReference();
		}
		if (m_data[index])
		{
			m_data[index]->removeReference();
		}
		m_data[index] = impl;
	}
	virtual void setSize( int n )
	{
		if (n < m_data.getSize())
		{
			for (int i = n; i < m_data.getSize(); i++)
			{
				hkDataArrayImpl* child = m_data[i];
				if (child)
				{
					child->removeReference();
				}
			}
		}

		Parent::setSize(n);
	}
	virtual hkDataArrayImpl* asArray(int i) const
	{
		hkDataArrayImpl* child = m_data[i];
		// Set up an array if there isn't one there
		if (!child)
		{
			hkArrayBase<hkDataArrayImpl*>& data = const_cast<hkArrayBase<hkDataArrayImpl*>&>(m_data);
			child = hkDataArrayDict_create( m_world, m_type->getParent(), 0 );
			child->addReference();
			data[i] = child;
		}
		return child;
	}
	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) { return HK_FAILURE; }
};

struct StructArrayImplementation : public hkDataArrayImpl
{
	typedef hkDataObject::MemberHandle MemberHandle;
	struct MemberData
	{
		InternedString m_memberName;
		hkDataArrayImpl* m_values;
	};
	typedef hkDataClassDict::MemberInfo MemberInfo;
	hkArrayBase<MemberData> m_memberData;
	hkRefPtr<const hkDataClassDict> m_class;
	hkDataObject::Type m_type;
	hkInt32 m_size;
	hkDataWorldDict* m_world;

	hkMemoryAllocator& getAllocator()
	{
		return *m_world->m_allocator;
	}

	inline InternedString intern(const char* s) const
	{
		return m_class->intern(s);
	}

	inline int _find(InternedString s) const
	{
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			if( m_memberData[i].m_memberName == s )
			{
				return i;
			}
		}
		return -1;
	}

	
	inline int _find(MemberHandle handle) const
	{
		const MemberInfo* info = (const MemberInfo*)handle;
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			if( m_memberData[i].m_memberName == info->m_name )
			{
				return i;
			}
		}
		return -1;
	}

	inline int _find(const char* nameIn) const
	{
		return _find( intern(nameIn) );
	}

	// return index of class member set in this array, may differ from class member index
	HK_NEVER_INLINE int _addMember(const hkDataClassDict::MemberInfo& mi, int size)
	{
		HK_ASSERT(0x2342a34a, m_world->m_typeManager.isOwned(mi.m_type));

		if( mi.m_type->isVoid())
		{
			return -1;
		}
		MemberData& md = m_memberData._expandOne(getAllocator());
		md.m_memberName = mi.m_name;

		hkDataObject::Type type = mi.m_type;
		hkDataObject::Type term = type->findTerminal();

		if (term->isClass() && term->getTypeName())
		{
			HK_ON_DEBUG(hkDataClassDict* dc = m_class->getWorld()->m_tracker->findTrackedClass(term->getTypeName()));
			HK_ASSERT(0x12e9cea0, dc);
		}
		
		md.m_values = hkDataArrayDict_create( m_world, type, 0 );
		md.m_values->addReference();
		md.m_values->setSize(size);

		if(mi.m_valuePtr || type->isClass())
		{
			for(int index = 0; index < size; index++)
			{
				if(type->isInt() || type->isByte())
				{
					md.m_values->setInt64(index, *reinterpret_cast<const hkInt64 *>(mi.m_valuePtr));
				}
				else if(type->isReal())
				{
					md.m_values->setReal(index, *reinterpret_cast<const hkReal *>(mi.m_valuePtr));
				}
				else if(type->isTuple())
				{
					if (type->getParent()->isReal())
					{
						md.m_values->setVec(index, reinterpret_cast<const hkReal*>(mi.m_valuePtr));
					}
					else
					{
						HK_ASSERT(0x23243242, !"Not sure how to set");
					}
				}
				else if(type->isClass())
				{
					// Create struct objects here so they can be
					// accessed. Defaults for members are automatically
					// handled by DataObjectDict
					// NOTE! This needs both the creation and setting of an object to work correctly.

					hkDataClassDict* dc = m_class->getWorld()->m_tracker->findTrackedClass(type->getTypeName());
					HK_ASSERT(0x3244323a, dc);

					hkDataObject impl(m_class->m_world->newObject(dc));
					md.m_values->setObject(index, impl.getImplementation());
					//hkDataObject obj = md.m_values->asObject(index);
				}
				else if(type->isCstring())
				{
					// setString will strDup() the string
					md.m_values->setString(index, reinterpret_cast<const char*>(mi.m_valuePtr));
				}				
				else
				{
					HK_ASSERT2(0x7c5717f0, 0, "Unsupported default value in dataobject");
				}
			}
		}
		return m_memberData.getSize()-1;
	}
	hkResult asStridedBasicArray(hkStridedBasicArray& arrOut) { return HK_FAILURE; }
#if 0
	inline int addMember(InternedString nameIn)
	{
		HK_ASSERT(0x44665f3e, _find( nameIn ) == -1);
		int i = m_class->getMemberIndexByName(nameIn.getString());
		HK_ASSERT3(0x2dbf1122, i != -1, "Class " << m_class->getName() << " does not have a member " << nameIn.getString() );

		int idx = _addMember(m_class->getMemberInfoByIndex(i), getSize());

		return idx;
	}

	HK_NEVER_INLINE int _findOrAdd(InternedString name)
	{
		int i = _find(name);
		if( i != -1 )
		{
			return i;
		}

		return addMember(name);
	}
#else
	inline int addMember(MemberHandle handle)
	{
		const MemberInfo* info = (const MemberInfo*)handle;
		return _addMember(*info, getSize());
	}

	HK_NEVER_INLINE int _findOrAdd(MemberHandle handle)
	{
		int i = _find(handle);
		if( i != -1 )
		{
			return i;
		}

		return addMember(handle);
	}
#endif

	inline void removeMember(InternedString s)
	{
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			MemberData& md = m_memberData[i];
			if( md.m_memberName == s )
			{
				md.m_values->removeReference();
				m_memberData.removeAtAndCopy(i);
				break;
			}
		}
	}

	inline void renameMember(InternedString oldName, InternedString newName)
	{
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			MemberData& md = m_memberData[i];
			if( md.m_memberName == oldName )
			{
				md.m_memberName = newName;
				break;
			}
		}
	}

	struct Object : public hkDataObjectImpl
	{
		StructArrayImplementation* m_impl;
		int m_index;

		Object(StructArrayImplementation* impl, int i) : m_impl(impl), m_index(i) {}

		virtual const hkDataClassImpl* getClass() const
		{
			return m_impl->getClass();
		}
		virtual hkDataObject::Value accessByName(const char* nameIn)
		{
			InternedString name = m_impl->intern(nameIn);
			const hkDataClassDict::MemberInfo* info = m_impl->m_class->getMemberInfoByName(name);
			
			if (info)
			{
				HK_ASSERT(0x3423432a, name.getString() == info->m_name.getString());
				return hkDataObject::Value(this, (MemberHandle)info);
			}
			else
			{
				return hkDataObject::Value(HK_NULL, HK_NULL);
			}
		}
		virtual hkBool32 hasMember(const char* name) const
		{
			InternedString internedMemberName = intern(name);
			for( int i = 0; i < m_impl->m_memberData.getSize(); ++i )
			{
				if( internedMemberName == m_impl->m_memberData[i].m_memberName )
				{
					return true;
				}
			}
			return false;
		}
		virtual hkBool32 isSet(MemberHandle handle)
		{
			const hkDataClassDict::MemberInfo* info = (const hkDataClassDict::MemberInfo*)handle;
			const hkArrayBase<MemberData>& memberData = m_impl->m_memberData;
			for( int i = 0; i < memberData.getSize(); ++i )
			{
				if( info->m_name == memberData[i].m_memberName )
				{
					return true;
				}
			}
			return false;
		}

		hkDataObject::Handle getHandle() const
		{
			hkDataObject::Handle h;
			h.p0 = const_cast<Object*>(this);
			h.p1 = 0;
			return h;
		}

		virtual Iterator getMemberIterator() const
		{
			return 0;
		}
		virtual hkBool32 isValid(Iterator it) const
		{
			return it < m_impl->m_memberData.getSize();
		}
		virtual Iterator getNextMember(Iterator it) const
		{
			return it+1;
		}
		virtual const char* getMemberName(Iterator it) const
		{
			return m_impl->m_memberData[it].m_memberName.getString();
		}
		virtual const hkDataObject::Value getMemberValue(Iterator it) const
		{
			//InternedString s; s.s = reinterpret_cast<const char*>( hkUlong(it) );
			return hkDataObject::Value( const_cast<Object*>(this), (MemberHandle)m_impl->m_memberData[it].m_memberName.getString() );
		}
		virtual void getAllMemberHandles(hkArrayBase<hkDataObject::MemberHandle>& handles) const
		{
			m_impl->m_class->_getAllMemberHandles(handles);
		}

		virtual InternedString intern(const char* name) const { return m_impl->intern(name); }
		virtual void selfDestruct() {}

			/// Doesn't actually do anything, because this is just a view on an array
		virtual void destroy() {}

		//virtual hkDataArrayImpl* asArray( const char* name ){ HK_ASSERT(0x52191e5c, 0); return HK_NULL; }
		virtual const hkReal* asVec( MemberHandle handle, int n )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			return m_impl->m_memberData[memIdx].m_values->asVec(m_index);
		}

#		define AS(HKTYPE, WHAT) \
			virtual HKTYPE as##WHAT( MemberHandle handle ) \
			{ \
				int memIdx = m_impl->_findOrAdd(handle); \
				return m_impl->m_memberData[memIdx].m_values->as##WHAT(m_index); \
			}
		AS(const char*, String)
		AS(hkDataArrayImpl*, Array)
		AS(hkInt64, Int)
		AS(hkInt64, Int64)
		AS(hkReal, Real)
		AS(hkHalf, Half)
#		undef AS

		virtual hkDataObjectImpl* asObject( MemberHandle handle )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			return m_impl->m_memberData[memIdx].m_values->asObject(m_index);
		}

		virtual void assign( MemberHandle handle, const hkDataObject::Value& valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x50319efd, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->set(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkDataObjectImpl* valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x50319efd, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setObject(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkDataArrayImpl* valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x6e34ce53, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setArray(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, const char* valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x28b4ef8a, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setString(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkReal valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x3f7b277f, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setReal(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkHalf valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x3f7b2770, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setHalf(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, const hkReal* valueIn, int nreal )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x4b5d69ce, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setVec(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkInt8 valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x464f1414, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setInt(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkInt16 valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x5e0b6cdc, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setInt(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkInt32 valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x64b21f5b, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setInt(m_index, valueIn);
		}
		virtual void assign( MemberHandle handle, hkInt64 valueIn )
		{
			int memIdx = m_impl->_findOrAdd(handle);
			HK_ASSERT(0x3751e6fa, memIdx != -1);
			m_impl->m_memberData[memIdx].m_values->setInt64(m_index, valueIn);
		}
		virtual void getMemberInfo(MemberHandle handle, hkDataClass::MemberInfo& infoOut)
		{
			const hkDataClassDict::MemberInfo* minfo = (const hkDataClassDict::MemberInfo*)handle;
			minfo->getMemberInfo(HK_NULL, infoOut);
		}

		virtual hkBool32 createdDuringPatching() const
		{
			return false;
		}
	};

	StructArrayImplementation(hkDataWorldDict* world, hkDataObject::Type type, const hkDataClassDict* klass, int size=0) : 
		m_class(klass), 
		m_type(type),
		m_size(size), 
		m_world(world) 
	{
		for( int i = 0; i < m_class->getNumMembers(); ++i )
		{
			_addMember(m_class->getMemberInfoByIndex(i), m_size);
		}
		m_world->m_tracker->trackStructArray(this);
	}
	~StructArrayImplementation()
	{
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			m_memberData[i].m_values->removeReference();
		}
		m_world->m_tracker->untrackStructArray(this);
		m_memberData._clearAndDeallocate(getAllocator());
	}

	virtual const hkDataClassImpl* getClass() const
	{
		return m_class;
	}
	virtual hkDataObjectImpl* asObject(int i) const
	{
		return new Object(const_cast<StructArrayImplementation*>(this),i);
	}
	virtual void setObject(int index, hkDataObjectImpl* val)
	{
		HK_ASSERT(0x36054842, 0 <= index && index < m_size);
		HK_ASSERT(0x34053172, val->getClass() == m_class);
		val->addReference();
		for( int memberIndex = 0; memberIndex < m_memberData.getSize(); ++memberIndex )
		{
			MemberData& md = m_memberData[memberIndex];
			md.m_values->set( index, val->accessByName(md.m_memberName.getString()) );
		}
		val->removeReference();
	}

	virtual hkDataObject::Type getType() const
	{
		return m_type;
	}
	virtual hkDataWorld* getWorld() const
	{
		return m_world;
	}
	virtual void clear()
	{
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			m_memberData[i].m_values->clear();
		}
		m_size = 0;
	}
	virtual void reserve( int n )
	{
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			m_memberData[i].m_values->reserve(n);
		}
	}
	virtual void setSize(int n)
	{
		m_size = n;
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			m_memberData[i].m_values->setSize(n);
		}
	}
	virtual int getSize() const
	{
#if defined(HK_DEBUG)
		int size = 0;
		for( int i = 0; i < m_memberData.getSize(); ++i )
		{
			int s = m_memberData[i].m_values->getSize();
			if( s )
			{
				HK_ASSERT(0x1a062752, size == 0 || size == s );
				size = s;
			}
		}
		HK_ASSERT(0x3f49089b, size == m_size);
#endif
		return m_size;
	}
	hkDataArrayImpl* swizzleObjectMember(const char* name) const
	{
		int i = _find(name);
		return m_memberData[i].m_values;
	}

};

hkDataArrayImpl* hkDataArrayDict_create( hkDataWorldDict* world, hkDataObject::Type t, int size )
{
	hkTypeManager::SubType subType = t->getSubType();

	switch (subType)
	{
		case hkTypeManager::SUB_TYPE_ARRAY:
		{
			ArrayArrayImplementation* array = new ArrayArrayImplementation(world, t);
			array->setSize(size);

			for (int i = 0; i < size; i++)
			{
				hkDataArrayImpl* subArray = hkDataArrayDict_create(world, t->getParent(), 0);
				array->setArray(i, subArray);
			}

			return array;
		}
		case hkTypeManager::SUB_TYPE_TUPLE:
		{
			if (t->getParent()->isReal())
			{
				// Vec types
				return new VecArrayImplementation(world, t);
			}

			// Recurse
			hkDataArrayImpl* sub = hkDataArrayDict_create( world, t->getParent(), 0 );
			return new ArrayOfTuplesImplementation( world, sub, t->getTupleSize());
		}
		case hkTypeManager::SUB_TYPE_INT:	
		{
			return new VariableIntArrayImplementation(world);
		}
		case hkTypeManager::SUB_TYPE_CSTRING:
		{
			return new CstringArrayImplementation(world, t);
		}
		case hkTypeManager::SUB_TYPE_REAL:
		{
			return new RealArrayImplementation(world, t);
		}
		case hkTypeManager::SUB_TYPE_BYTE:
		{
			return new ByteArrayImplementation(world, t);
		}
		case hkTypeManager::SUB_TYPE_POINTER:
		{
			hkDataObject::Type parent = t->getParent(); 
			HK_ASSERT(0x23432, parent->isClass());

			if (parent->getTypeName())
			{
				// Look up the class
				hkDataClassImpl* cls = world->findClass(parent->getTypeName());
				if (!cls)
				{
					HK_WARN(0x2432a4aa, "Couldn't find class '" << parent->getTypeName() << "'");
					return HK_NULL;
				}
			}
			return new PointerArrayImplementation(world, t);
		}
		case hkTypeManager::SUB_TYPE_CLASS:
		{
			// Look up the class
			hkDataClassImpl* cls = world->findClass(t->getTypeName());
			if (!cls)
			{
				HK_WARN(0x2432a4aa, "Couldn't find class '" << t->getTypeName() << "'");
				return HK_NULL;
			}
			return new StructArrayImplementation(world, t, static_cast<hkDataClassDict*>(cls), size);
		}
		default: 
		{
			HK_ASSERT(0x234324, !"Unhandled type");
		}
	}

	return HK_NULL;
}


//////////////////////////////////////////////////////////////////////////
// WorldDict
//////////////////////////////////////////////////////////////////////////

hkDataWorldDict::ObjectTracker::ObjectTracker(hkMemoryAllocator* mall) : m_allocator(mall), m_topLevelObject(HK_NULL)
{

}

hkDataWorldDict::ObjectTracker::~ObjectTracker()
{
	hkLocalArray<hkDataObjectDict*> objs( m_objectsFromClass.m_valueChain.getSize() ); // untrackObject() mutates map so copy
	hkLocalArray<hkDataClassDict*> classes( m_classes.getSize() ); // untrackClass() mutates map so copy
	m_derivedFromParent.clear();
	m_topLevelObject = HK_NULL;

	for( hkStringMap<hkDataClassDict*>::Iterator it = m_classes.getIterator(); m_classes.isValid(it); it = m_classes.getNext(it) )
	{
		hkDataClassDict* c = m_classes.getValue(it);
		classes.pushBackUnchecked(c);
		int i = m_objectsFromClass.getFirstIndex(c);
		if( i != -1 )
		{
			for( ; i != -1; i = m_objectsFromClass.getNextIndex(i) )
			{
				hkDataObjectDict* obj = m_objectsFromClass.getValue(i);
				HK_ASSERT(0x150c822b, obj);
				objs.pushBack(obj);
			}
			m_objectsFromClass.removeKey(c);
		}
	}
	for( int i = 0; i < objs.getSize(); ++i )
	{
		hkDataObjectDict* obj = objs[i];

		if( obj->getExternalReferenceCount() > 0 )
		{
			HK_WARN_ALWAYS(0x760d3795, "The object of class " << obj->getClass()->getName() << " is about to be removed and all references to it will be invalid.\n"\
				"However, the object is referenced (" << obj->getExternalReferenceCount() << ") from outside the world and it will lead to unexpected behavior or crash.");
		}

		obj->selfDestruct();
	}
	for( int i = 0; i < objs.getSize(); ++i )
	{
		objs[i]->removeReference();
	}

	for( int i = 0; i < classes.getSize(); ++i )
	{
		classes[i]->removeReference();
	}

	for( hkStringMap<const char*>::Iterator it = m_internedClassNames.getIterator(); m_internedClassNames.isValid(it); it = m_internedClassNames.getNext(it) )
	{
		m_internedClassNames.getValue(it)->removeReference();
	}

	for( hkStringMap<const char*>::Iterator it = m_interns.getIterator(); m_interns.isValid(it); it = m_interns.getNext(it) )
	{
		hkString::strFree( m_interns.getValue(it) );
	}

	{
		ClassToArraysMap::Iterator iter = m_arraysFromClass.getIterator();
		for (; m_arraysFromClass.isValid(iter); iter = m_arraysFromClass.getNext(iter))
		{
			ArrayMap* map = m_arraysFromClass.getValue(iter);
			delete map;
		}
	}
}

inline void hkDataWorldDict::ObjectTracker::trackObject(hkDataObjectDict* obj)
{
	hkDataClassDict* c = static_cast<hkDataClassDict*>(const_cast<hkDataClassImpl*>(obj->getClass()));
	HK_ASSERT(0x378deb57, findTrackedClass(obj->getClass()->getName()));
	m_objectsFromClass.insert(c, obj);
	if( m_topLevelObject == HK_NULL )
	{
		m_topLevelObject = obj;
	}
}

inline void hkDataWorldDict::ObjectTracker::trackStructArray(hkDataArrayImpl* a)
{
	hkDataClassDict* c = static_cast<hkDataClassDict*>(const_cast<hkDataClassImpl*>(a->getClass()));


	ArrayMap* map = m_arraysFromClass.getWithDefault(c, HK_NULL);
	if (!map)
	{
		map = new ArrayMap;
		m_arraysFromClass.insert(c, map);
	}
	HK_ASSERT(0x3244324, !map->hasKey(a));
	map->insert(a, 0);	
}

inline void hkDataWorldDict::ObjectTracker::untrackStructArray(hkDataArrayImpl* a)
{
	hkDataClassDict* c = static_cast<hkDataClassDict*>(const_cast<hkDataClassImpl*>(a->getClass()));

	ArrayMap* map = m_arraysFromClass.getWithDefault(c, HK_NULL);
	HK_ASSERT(0x2432a432, map);
	map->remove(a);
}

inline void hkDataWorldDict::ObjectTracker::getTrackedStructArrays(hkDataClassDict* c, hkBool32 baseClass, hkArray<hkDataArrayImpl*>::Temp& arraysOut)
{
	{
		ArrayMap* map = m_arraysFromClass.getWithDefault(c, HK_NULL);
		if (map && map->getSize() > 0)
		{
			hkDataArrayImpl** dst = arraysOut.expandBy(map->getSize());
			ArrayMap::Iterator iter = map->getIterator();
			for (; map->isValid(iter); iter = map->getNext(iter))
			{
				*dst++ = map->getKey(iter);
			}
		}
	}

	if( baseClass )
	{
		for( int i = m_derivedFromParent.getFirstIndex(c); i != -1; i = m_derivedFromParent.getNextIndex(i) )
		{
			getTrackedStructArrays(m_derivedFromParent.getValue(i), baseClass, arraysOut);
		}
	}
}

inline void hkDataWorldDict::ObjectTracker::trackClass(hkDataClassDict* klass)
{
	hkDataClassDict* parent = static_cast<hkDataClassDict*>(klass->getParent());
	HK_ASSERT(0x68c0f9b2, !parent || m_classes.hasKey(parent->getName()));
	if( m_classes.hasKey(klass->getName()) == false )
	{
		m_classes.insert(klass->getName(), klass);
		if( parent )
		{
#if defined(HK_DEBUG)
			for( int i = m_derivedFromParent.getFirstIndex(parent); i != -1; i = m_derivedFromParent.getNextIndex(i) )
			{
				HK_ASSERT(0x6cdc47ea, m_derivedFromParent.getValue(i) != klass);
			}
#endif
			m_derivedFromParent.insert(parent, klass);
		}
	}
	else
	{
		HK_ASSERT(0x2bb058f7, findTrackedClass(klass->getName())->getVersion() == klass->getVersion());
	}
}

inline void hkDataWorldDict::ObjectTracker::findTrackedClasses(hkArray<hkDataClassImpl*>::Temp& classesOut) const
{
	classesOut.reserve(m_classes.getSize());
	for( hkStringMap<hkDataClassDict*>::Iterator it = m_classes.getIterator(); m_classes.isValid(it); it = m_classes.getNext(it) )
	{
		hkDataClassDict* c = m_classes.getValue(it);
		classesOut.pushBackUnchecked(c);
	}
}

inline void hkDataWorldDict::ObjectTracker::findTrackedObjectsByBase(const char* className, hkBool32 baseClass, hkBool32 addStructs, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const
{
	hkDataClassDict* c = findTrackedClass(className);
	if( c )
	{
		for( int i = m_objectsFromClass.getFirstIndex(c); i != -1; i = m_objectsFromClass.getNextIndex(i) )
		{
			objectsOut.pushBack(m_objectsFromClass.getValue(i));
		}
		if( addStructs )
		{

			ArrayMap* map = m_arraysFromClass.getWithDefault(c, HK_NULL);
			if (map && map->getSize() > 0)
			{
				ArrayMap::Iterator iter = map->getIterator();
				for (; map->isValid(iter); iter = map->getNext(iter))
				{
					hkDataArrayImpl* a = map->getKey(iter);
					
					for( int j = 0; j < a->getSize(); ++j )
					{
						objectsOut.pushBack(a->asObject(j));
					}
				}
			}
		}
		if( baseClass )
		{
			for( int i = m_derivedFromParent.getFirstIndex(c); i != -1; i = m_derivedFromParent.getNextIndex(i) )
			{
				findTrackedObjectsByBase(m_derivedFromParent.getValue(i)->getName(), baseClass, addStructs, objectsOut);
			}
		}
	}
}

inline void hkDataWorldDict::ObjectTracker::getTrackedObjects(const char* className, hkBool32 baseClass, hkBool32 addStructs, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const
{
	objectsOut.clear();
	findTrackedObjectsByBase(className, baseClass, addStructs, objectsOut);
}

inline void hkDataWorldDict::ObjectTracker::untrackObject(hkDataObjectDict* obj)
{
	if( m_topLevelObject == obj )
	{
		m_topLevelObject = HK_NULL;
	}

	m_objectsFromClass.removeByValue(static_cast<hkDataClassDict*>(const_cast<hkDataClassImpl*>(obj->getClass())), obj);
}

inline void hkDataWorldDict::ObjectTracker::removeClassContent(hkDataClassDict* klass)
{
	const char* className = klass->getName();
	hkDataClassDict* c = findTrackedClass(className);
	HK_ASSERT(0x35167233, c && c == klass);

	// remove derived class contents now
	for( int i = m_derivedFromParent.getFirstIndex(klass); i != -1; i = m_derivedFromParent.getNextIndex(i) )
	{
		hkDataClassDict* derived = m_derivedFromParent.getValue(i);
		HK_ASSERT(0x24585fd2, m_classes.hasKey(derived->getName()));
		removeClassContent(derived);
	}

	// remove class objects
	int idx = m_objectsFromClass.getFirstIndex(c);
	if( idx != -1 )
	{
		hkLocalArray<hkDataObjectDict*> objs(512); //objectDestroyed callback mutates map so copy
		for( ; idx != -1; idx = m_objectsFromClass.getNextIndex(idx) )
		{
			hkDataObjectDict* obj = m_objectsFromClass.getValue(idx);
			HK_ASSERT(0x150c822b, obj);
			objs.pushBack(obj);
		}
		for( int i = 0; i < objs.getSize(); ++i )
		{
			if( m_topLevelObject == objs[i] )
			{
				m_topLevelObject = HK_NULL;
			}
			if( objs[i]->getExternalReferenceCount() > 0 )
			{
				HK_WARN_ALWAYS(0x760d3794, "The object of class " << objs[i]->getClass()->getName() << " is about to be removed and all references to it will be invalid.\n"\
					"However, the object is referenced (" << objs[i]->getExternalReferenceCount() << ") from outside the world and it will lead to unexpected behavior or crash.");
			}
			objs[i]->selfDestruct();
		}
		for( int i = 0; i < objs.getSize(); ++i )
		{
			objs[i]->removeReference();
		}
		m_objectsFromClass.removeKey(c);
	}
}

inline void hkDataWorldDict::ObjectTracker::untrackClass(hkDataClassDict* klass)
{
	const char* className = klass->getName();
	hkDataClassDict* c = findTrackedClass(className);
	HK_ASSERT(0x35167233, c && c == klass);

	removeClassContent(klass);

	if( hkDataClassDict* parent = static_cast<hkDataClassDict*>(c->getParent()) )
	{
		// untrack from parent's list first
		m_derivedFromParent.removeByValue(parent, klass);
	}

	klass->selfDestruct();

	{
		// untrack derived classes now
		hkLocalArray<hkDataClassDict*> classes(64);
		int i = m_derivedFromParent.getFirstIndex(klass);
		bool foundDerivedClasses = i != -1;
		if( foundDerivedClasses )
		{
			for( ; i != -1; i = m_derivedFromParent.getNextIndex(i) )
			{
				hkDataClassDict* derived = m_derivedFromParent.getValue(i);
				HK_ASSERT(0x24585fd2, m_classes.hasKey(derived->getName()));
				classes.pushBack(derived);
			}
		}
		for( i = 0; i < classes.getSize(); ++i )
		{
			untrackClass(classes[i]);
			classes[i]->removeReference();
		}
		if( foundDerivedClasses )
		{
			HK_ASSERT(0x22d91b6a, m_derivedFromParent.getFirstIndex(klass) == -1);
			m_derivedFromParent.removeKey(klass);
		}
	}

	InternedStringRefCounted* internedName = m_internedClassNames.getWithDefault(className, HK_NULL);
	HK_ASSERT(0x588044c1, internedName);
	HK_ASSERT3(0x506a9c53, internedName->getReferenceCount() == 1,
		"The class name '" << className << "' has " << (internedName->getReferenceCount()-1) << " external reference(s).");
#if 1
	m_internedClassNames.remove(className);
	internedName->removeReference();
#else
	if( internedName->getReferenceCount() == 1 )
	{
		m_internedClassNames.remove(className);
		internedName->removeReference();
	}
	else
	{
		HK_WARN_ALWAYS(0x506a9c53,
			"The class name '" << className << "' has " << (internedName->getReferenceCount()-1) << " external reference(s).");
	}
#endif
	m_classes.remove(className);
}

inline void hkDataWorldDict::ObjectTracker::retrackDerivedClass(hkDataClassDict* oldParent, hkDataClassDict* klass)
{
	if( oldParent )
	{
		HK_ASSERT(0x5184302f, m_classes.hasKey(oldParent->getName()));
		HK_ON_DEBUG(bool found = false);
		for( int i = m_derivedFromParent.getFirstIndex(oldParent); i != -1; i = m_derivedFromParent.getNextIndex(i) )
		{
			if( m_derivedFromParent.getValue(i) == klass )
			{
				m_derivedFromParent.removeByIndex(oldParent, i);
				HK_ON_DEBUG(found = true);
				break;
			}
		}
		HK_ASSERT(0x523caf26, found);
	}
	hkDataClassDict* newParent = static_cast<hkDataClassDict*>(klass->getParent());
	if( newParent )
	{
		m_derivedFromParent.insert(newParent, klass);
	}
}

inline void hkDataWorldDict::ObjectTracker::retrackRenamedClass(const char* oldName, const char* newName)
{
	hkDataClassDict* c = findTrackedClass(oldName);
	HK_ASSERT(0x51889343, findTrackedClass(oldName) );
	HK_ASSERT(0x41462f60, findTrackedClass(newName) == HK_NULL );
	const char* newNameIntern = intern(newName).getString();
	replaceClassNameInInternHandle(oldName, newNameIntern);
	m_classes.remove(oldName);
	HK_ASSERT(0x3e0f64cc, hkString::strCmp(newName, c->getName()) == 0);
	m_classes.insert(newNameIntern, c);
}

inline hkDataObjectDict* hkDataWorldDict::ObjectTracker::getTopLevelObject()
{
	return m_topLevelObject;
}

inline void hkDataWorldDict::ObjectTracker::retractCastedObject(hkDataObjectDict* obj, hkDataClassDict* newClass)
{
	HK_ASSERT(0x339215d5, findTrackedClass(obj->getClass()->getName()) );
	HK_ASSERT(0x2b1c9763, findTrackedClass(newClass->getName()) );
	hkDataClassDict* c = static_cast<hkDataClassDict*>(const_cast<hkDataClassImpl*>(obj->getClass()));
	for( int i = m_objectsFromClass.getFirstIndex(c); c != HK_NULL && i != -1; i = m_objectsFromClass.getNextIndex(i) )
	{
		hkDataObjectDict* o = m_objectsFromClass.getValue(i);
		HK_ASSERT(0x150c822b, o);
		if( obj == o )
		{
			m_objectsFromClass.removeByIndex(c, i);
			m_objectsFromClass.insert(newClass, obj);
			c = HK_NULL;
		}
	}
	HK_ASSERT(0x59bde522, c == HK_NULL);
}

inline void hkDataWorldDict::ObjectTracker::retractCastedObjects(hkDataClassDict* oldClass, hkDataClassDict* newClass)
{
	HK_ASSERT(0x339215d5, findTrackedClass(oldClass->getName()) );
	HK_ASSERT(0x2b1c9763, findTrackedClass(newClass->getName()) );
	m_objectsFromClass.changeKey(oldClass, newClass);
}

inline InternedString hkDataWorldDict::ObjectTracker::intern(const char* sin)
{
	if( sin != HK_NULL )
	{
		char* s = m_interns.getWithDefault(sin, HK_NULL);
		if( s == HK_NULL )
		{
			s = hkString::strDup(sin);
			m_interns.insert(s,s);
		}
		return InternedString(s);
	}
	else
	{
		return InternedString();
	}
}

inline InternedStringHandle hkDataWorldDict::ObjectTracker::internClassNameHandle(const char* sin)
{
	InternedStringHandle nameHandle;
	if( sin )
	{
		InternedStringRefCounted* interned = m_internedClassNames.getWithDefault(sin, HK_NULL);
		if( interned == HK_NULL )
		{
			interned = new InternedStringRefCounted(intern(sin));
			m_internedClassNames.insert(interned->getString(), interned);
		}
		nameHandle.m_cached = interned;
	}
	return nameHandle;
}

inline void hkDataWorldDict::ObjectTracker::replaceClassNameInInternHandle(const char* oldName, const char* newName)
{
	HK_ASSERT(0x12bc4e82, oldName && newName);
	HK_ASSERT(0x69fd96e0, intern(newName).getString() == newName);
	if( oldName != newName )
	{
		InternedStringRefCounted* interned = m_internedClassNames.getWithDefault(oldName, HK_NULL);
		HK_ASSERT(0x6c0631ae, interned);
		HK_ASSERT3(0x190bdbe4, m_internedClassNames.getWithDefault(newName, HK_NULL) == HK_NULL,
			"Cannot rename '" << oldName << "' to '" << newName << "'.\nThe class name '" << newName << "' is still in use.");
		m_internedClassNames.remove(oldName);
		interned->m_cachedString = newName;
		m_internedClassNames.insert(newName, interned);
	}
}

hkDataWorldDict::hkDataWorldDict()
{
	m_allocator = &hkMemoryRouter::getInstance().heap();
	m_tracker = new ObjectTracker(m_allocator);
}

hkDataWorldDict::hkDataWorldDict(hkMemoryAllocator* mall)
{
	m_allocator = mall;
	m_tracker = new ObjectTracker(m_allocator);
}

hkDataWorldDict::~hkDataWorldDict()
{
	delete m_tracker;
}

hkDataObject hkDataWorldDict::getContents() const
{
	return m_tracker->getTopLevelObject();
}

hkDataObjectImpl* hkDataWorldDict::newObject(const hkDataClass& klass, bool createdDuringPatching) const
{
	hkDataObjectDict* d = new hkDataObjectDict(static_cast<const hkDataClassDict*>(klass.getImplementation()), createdDuringPatching);
	d->addReference();
	m_tracker->trackObject(d);
	return d;
}

hkDataClassImpl* hkDataWorldDict::newClass(const hkDataClass::Cinfo& cinfo)
{

	HK_ASSERT(0x1171a8e8, m_tracker->findTrackedClass(cinfo.name) == HK_NULL);
	hkDataClassDict* c = new hkDataClassDict(this, cinfo.name, cinfo.version);

	if( cinfo.parent )
	{
		hkDataClassImpl* parent = m_tracker->findTrackedClass(cinfo.parent);
		HK_ASSERT(0x1171a8a9, parent != HK_NULL);
		c->setParent( static_cast<hkDataClassDict*>(parent) );
	}
	c->addReference();
	m_tracker->trackClass(c);

	for( int i = 0; i < cinfo.members.getSize(); ++i )
	{
		const hkDataClass::Cinfo::Member& m = cinfo.members[i];
		HK_ASSERT(0x23432432, m_typeManager.isOwned(m.type));
		c->addMember( m_tracker->intern(m.name), m.type, HK_NULL); //TODO typename
	}

	// Add the type name to the type manager
	m_typeManager.addClass(cinfo.name);

	return c;
}

hkDataArrayImpl* hkDataWorldDict::newArray(hkDataObject& obj, hkDataObject::MemberHandle handle, const hkDataClass::MemberInfo& minfo) const
{
	hkDataWorldDict* nonConstThis = const_cast<hkDataWorldDict*>(this);
	hkDataArrayImpl* ret;
	hkDataObject::Type type = minfo.m_type;
	if (type->isTuple())
	{
		const int tupleSize = type->getTupleSize();
		ret = hkDataArrayDict_create( nonConstThis, type->getParent(), tupleSize);
		//ret->setSize(tupleSize);
	}
	else
	{
		ret = hkDataArrayDict_create( nonConstThis, type->getParent(), 0);
	}

	hkDataObject::Value value(obj.getImplementation(), handle);
	value = hkDataArray(ret);

	//obj[minfo.m_name] = hkDataArray(ret);
	return ret;
}

hkEnum<hkDataWorld::DataWorldType, hkInt32> hkDataWorldDict::getType() const
{
	return TYPE_DICTIONARY;
}

// manage classes
void hkDataWorldDict::findAllClasses(hkArray<hkDataClassImpl*>::Temp& classesOut) const
{
	m_tracker->findTrackedClasses(classesOut);
}

hkDataClassImpl* hkDataWorldDict::wrapClass(const hkClass& klass)
{
	HK_ASSERT(0x1171a8e9, m_tracker->findTrackedClass(klass.getName()) == HK_NULL);
	hkDataClassDict* c = new hkDataClassDict(this, klass.getName(), klass.getDescribedVersion());

	if( klass.getParent() )
	{
		hkDataClassDict* p = m_tracker->findTrackedClass(klass.getParent()->getName());
		if( p == HK_NULL )
		{
			p = static_cast<hkDataClassDict*>(wrapClass(*klass.getParent()));
		}
		HK_ASSERT(0x29a593fc, p);
		c->setParent( p );
	}
	c->addReference();
	m_tracker->trackClass(c);

	for( int i = 0; i < klass.getNumDeclaredMembers(); ++i )
	{
		const hkClassMember& m = klass.getDeclaredMember(i);
		hkDataObject::Type mtype = m_typeManager.getSubType(hkTypeManager::SUB_TYPE_VOID);
		
		if( !m.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) )
		{
			const hkClass* cls =  m.getClass();
			const char* typeName = HK_NULL;

			if( cls )
			{
				typeName = cls->getName();

				if( const hkVariant* typeAttr = m.getAttribute(s_hkDataObjectTypeAttributeID) )
				{
					const hkClassMemberAccessor attrTypeName(*typeAttr, "typeName");
					typeName = attrTypeName.asCstring();
				}
				else if( const hkVariant* classAttr = cls->getAttribute(s_hkDataObjectTypeAttributeID) )
				{
					const hkClassMemberAccessor attrTypeName(*classAttr, "typeName");

					typeName = attrTypeName.asCstring();
				}
			}

			mtype = hkDataObjectUtil::getTypeFromMemberTypeClassName(m_typeManager, m.getType(), m.getSubType(), typeName, m.getCstyleArraySize());
		}

		c->addMember( m_tracker->intern(m.getName()), mtype, HK_NULL);
	}
	return c;
}

hkDataClassImpl* hkDataWorldDict::copyClassFromWorld(const char* name, const hkDataWorld& worldFrom)
{
	hkDataClassImpl* copied = findClass(name);
	if( copied )
	{
		return copied;
	}
	hkDataClass c = worldFrom.findClass(name);
	HK_ASSERT(0x30a91877, !c.isNull());
	hkDataClass::Cinfo cinfo;
	cinfo.name = c.getName();
	cinfo.version = c.getVersion();
	hkDataClass parent = c.getParent();
	if( parent.isNull() )
	{
		cinfo.parent = HK_NULL;
	}
	else
	{
		cinfo.parent = parent.getName();
		copied = findClass(cinfo.name);
		if( copied )
		{
			return copied;
		}
	}

	copied = newClass(cinfo);
	HK_ASSERT(0x413a06e1, copied);
	hkArray<hkDataClass::MemberInfo>::Temp members(c.getNumDeclaredMembers());
	c.getAllDeclaredMemberInfo(members);
	cinfo.members.reserve(members.getSize());
	hkDataClass classToAddMembers(copied);
	for( int i = 0; i < members.getSize(); ++i )
	{
		hkDataClass::MemberInfo& minfoFrom = members[i];

		hkDataObject::Type dstType = m_typeManager.copyType(minfoFrom.m_type);

		hkDataObject::Type terminalType = dstType->findTerminal();
		if (terminalType->isClass() && worldFrom.findClass(dstType->getTypeName()))
		{
			copyClassFromWorld(terminalType->getTypeName(), worldFrom );
		}
		
		addClassMember(classToAddMembers, minfoFrom.m_name, dstType, HK_NULL);
	}
	return copied;
}

hkDataClassImpl* hkDataWorldDict::findClass(const char* name) const
{
	return m_tracker->findTrackedClass(name);
}

hkDataObject hkDataWorldDict::findObject(const hkDataObject::Handle& handle) const
{
	return static_cast<hkDataObjectDict*>(handle.p0);
}

void hkDataWorldDict::renameClass(hkDataClass& klass, const char* newName)
{
	InternedString oldName = m_tracker->intern(klass.getName());
	HK_ON_DEBUG(hkDataClassImpl* cls = m_tracker->findTrackedClass(oldName.getString()));
	HK_ASSERT(0x33441d9a, cls);

	hkTypeManager::Type* type = m_typeManager.getClass(oldName.getString());
	if(type)
	{
		// Rename the class in the type manager
		m_typeManager.renameClass(oldName.getString(), newName);
	}

	m_tracker->retrackRenamedClass(oldName.getString(), newName);
}

void hkDataWorldDict::removeClass(hkDataClass& klass)
{
	HK_ASSERT(0x33441d9b, m_tracker->findTrackedClass(klass.getName()));

	hkTypeManager::Type* type = m_typeManager.getClass(klass.getName());
	if (type)
	{
		m_typeManager.removeClass(type);
	}

	hkDataClassDict* c = static_cast<hkDataClassDict*>(klass.getImplementation());
	klass = HK_NULL;
	m_tracker->untrackClass(c);
	c->removeReference();
}

void hkDataWorldDict::setClassVersion(hkDataClass& klass, int newVersion)
{
	HK_ASSERT(0x33441d9c, m_tracker->findTrackedClass(klass.getName()));
	static_cast<hkDataClassDict*>(klass.getImplementation())->setVersion(newVersion);
}

void hkDataWorldDict::setClassParent(hkDataClass& klass, hkDataClass& parent)
{
	HK_ASSERT(0x33441d9d, m_tracker->findTrackedClass(klass.getName()));
	hkDataClassDict* oldParent = static_cast<hkDataClassDict*>(klass.getImplementation()->getParent());
	HK_ASSERT(0x1230d0a7, !oldParent || m_tracker->findTrackedClass(oldParent->getName()));
	static_cast<hkDataClassDict*>(klass.getImplementation())->setParent(static_cast<hkDataClassDict*>(parent.getImplementation()));

	m_tracker->retrackDerivedClass(oldParent, static_cast<hkDataClassDict*>(klass.getImplementation()));
}

void hkDataWorldDict::addClassMember(hkDataClass& klass, const char* name, hkDataObject::Type type, const void* valuePtr)
{
	HK_ASSERT(0x34233a32, m_typeManager.isOwned(type));

	HK_ASSERT(0x33441d9e, m_tracker->findTrackedClass(klass.getName()));
	hkDataClassDict* c = static_cast<hkDataClassDict*>(klass.getImplementation());
	//HK_ASSERT(0x5e2ce30c, c->getMemberIndexByName(name) == -1); //
	InternedString internedName = m_tracker->intern(name);
	if( c->getDeclaredMemberIndexByName(name) != -1 )
	{
		c->removeMember(internedName);
	}
	c->addMember(internedName, type, valuePtr);
}

void hkDataWorldDict::setClassMemberDefault(hkDataClass& klass, const char* name, const void* valuePtr)
{
	HK_ASSERT(0x355bdbb4, m_tracker->findTrackedClass(klass.getName()));
	hkDataClassDict* c = static_cast<hkDataClassDict*>(klass.getImplementation());
	InternedString internedName = m_tracker->intern(name);
	HK_ASSERT2(0x66562d34, c->getDeclaredMemberIndexByName(name) != -1, "Patch attempts to set default for invalid member");
	c->setMemberDefault(internedName, valuePtr);
}


void hkDataWorldDict::renameClassMember(hkDataClass& klass, const char* oldName, const char* newName)
{
	HK_ASSERT(0x33441d9f, m_tracker->findTrackedClass(klass.getName()));
	hkDataClassDict* c = static_cast<hkDataClassDict*>(klass.getImplementation());
	HK_ASSERT(0x5e2ce30d, c->getMemberIndexByName(oldName) != -1);
	InternedString internedOldName = m_tracker->intern(oldName);
	InternedString internedNewName = m_tracker->intern(newName);
	hkArray<hkDataObjectImpl*>::Temp objs;
	m_tracker->getTrackedObjects(klass.getName(), true, false, objs);
	for( int index = 0; index < objs.getSize(); ++index )
	{
		static_cast<hkDataObjectDict*>(objs[index])->renameMember(internedOldName, internedNewName);
	}
	// arrays of structs
	hkArray<hkDataArrayImpl*>::Temp arrays;
	m_tracker->getTrackedStructArrays(c, true, arrays);
	for( int i = 0; i < arrays.getSize(); ++i )
	{
		static_cast<StructArrayImplementation*>(arrays[i])->renameMember(internedOldName, internedNewName);
	}
	c->renameMember(internedOldName, internedNewName);
}

void hkDataWorldDict::removeClassMember(hkDataClass& klass, const char* name)
{
	HK_ASSERT(0x33441e02, m_tracker->findTrackedClass(klass.getName()));
	hkDataClassDict* c = static_cast<hkDataClassDict*>(klass.getImplementation());
	HK_ASSERT(0x5e2ce30f, c->getMemberIndexByName(name) != -1);
	InternedString internedName = m_tracker->intern(name);
	hkArray<hkDataObjectImpl*>::Temp objs;
	m_tracker->getTrackedObjects(klass.getName(), true, false, objs);
	for( int index = 0; index < objs.getSize(); ++index )
	{
		static_cast<hkDataObjectDict*>(objs[index])->removeMember(internedName);
	}
	// arrays of structs
	hkArray<hkDataArrayImpl*>::Temp arrays;
	m_tracker->getTrackedStructArrays(c, true, arrays);
	for( int i = 0; i < arrays.getSize(); ++i )
	{
		static_cast<StructArrayImplementation*>(arrays[i])->removeMember(internedName);
	}
	c->removeMember(internedName);
}

// manage objects
void hkDataWorldDict::castObject(hkDataObject& obj, const hkDataClass& castClass)
{
	HK_ASSERT(0x33441e03, m_tracker->findTrackedClass(obj.getClass().getName()));
	HK_ASSERT(0x33441e04, m_tracker->findTrackedClass(castClass.getName()));
	hkDataObjectDict* o = static_cast<hkDataObjectDict*>(obj.getImplementation());
	hkDataClassDict* c = static_cast<hkDataClassDict*>(const_cast<hkDataClassImpl*>(castClass.getImplementation()));
	m_tracker->retractCastedObject(o, c);
}

void hkDataWorldDict::findObjectsByExactClass(const char* className, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const
{
	m_tracker->getTrackedObjects(className, false, true, objectsOut);
}

void hkDataWorldDict::findObjectsByBaseClass(const char* className, hkArray<hkDataObjectImpl*>::Temp& objectsOut) const
{
	m_tracker->getTrackedObjects(className, true, true, objectsOut);
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
