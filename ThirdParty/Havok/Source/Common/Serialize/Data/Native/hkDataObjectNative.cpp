/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Base/Container/RelArray/hkRelArray.h>
#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Base/Reflection/Registry/hkTypeInfoRegistry.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/Base/Reflection/Util/hkVariantDataUtil.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Data/Util/hkDataObjectUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

#include <Common/Base/Reflection/hkClassMemberAccessor.h>
#include <Common/Base/Types/hkTypedUnion.h>

class hkDataObjectNative;
class hkDataClassNative;
class hkDataArrayNative;

// todo: check all int types covered in switches

static const char* s_hkDataObjectTupeAttributeID = "hk.DataObjectType";

static hkDataObjectImpl* hkNativeDataObjectImplCreate(hkVariant& v, hkDataWorldNative* w, hkBool isRefOwned = false);

class hkDataClassNative : public hkDataClassImpl
{
	public:

		hkDataClassNative(const hkClass* k, hkDataWorldNative* w)
			: m_class(k)
			, m_world(w)
		{
			HK_ASSERT(0x5a8f1a88, hkUlong(k) > 0x1000 );
		}
		virtual const hkDataWorld* getWorld() const
		{
			return m_world;
		}

		virtual const char* getName() const
		{
			return m_class->getName();
		}
		virtual int getVersion() const
		{
			return m_class->getDescribedVersion();
		}
		virtual hkDataClassImpl* getParent() const
		{
			const hkClass* p = m_class->getParent();
			return p ? m_world->findClass(p->getName()) : HK_NULL;
		}
		virtual hkBool isSuperClass(const hkDataClassImpl* k) const
		{
			const hkDataClassImpl* c = k;
			while( c )
			{
				if( hkString::strCmp(c->getName(), getName()) == 0 )
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
			return m_class->getNumDeclaredMembers();
		}
		virtual int getDeclaredMemberIndexByName(const char* name) const
		{
			return m_class->getDeclaredMemberIndexByName(name);
		}

		// all members
		virtual int getNumMembers() const
		{
			return m_class->getNumMembers();
		}
		virtual int getMemberIndexByName(const char* name) const
		{
			return m_class->getMemberIndexByName(name);
		}
		static void _fillMemberInfo( const hkDataClassImpl* clsIn, hkDataWorldNative* world, const hkClassMember& mem, hkDataClass::MemberInfo& info ) 
		{
			hkTypeManager& typeManager = world->getTypeManager();

			info.m_name = mem.getName();
			info.m_owner = clsIn;
			if( mem.getFlags().allAreSet(hkClassMember::SERIALIZE_IGNORED) && (!world->m_accessSerializeIgnored))
			{
				info.m_type = typeManager.getSubType(hkTypeManager::SUB_TYPE_VOID);
			}
			else
			{
				const hkClass* cls = mem.getClass();
				if( cls )
				{
					if( const hkVariant* typeAttr = mem.getAttribute(s_hkDataObjectTupeAttributeID) )
					{
						const hkClassMemberAccessor attrTypeName(*typeAttr, "typeName");

						cls = world->m_reg->getClassByName(attrTypeName.asCstring());
						HK_ASSERT(0x342343, cls);
					}
					else if (hkString::strCmp(cls->getName(), "hkpMaxSizeMotion") == 0)
					{
						cls = world->m_reg->getClassByName("hkpMotion");
						HK_ASSERT(0x3423432a, cls);
					}

					// Make sure it exists
					HK_ASSERT(0x3423432, world->findClass(cls->getName()));
				}
				
				info.m_type = world->getTypeFromMemberType(mem.getType(), mem.getSubType(), cls, mem.getCstyleArraySize());
			}
		}

		HK_FORCE_INLINE void fillMemberInfo( const hkClassMember& mem, hkDataClass::MemberInfo& info ) const
		{
			_fillMemberInfo(this, m_world, mem, info);
		}
		

		virtual void getDeclaredMemberInfo(int i, hkDataClass::MemberInfo& info) const
		{
			fillMemberInfo( m_class->getDeclaredMember(i), info );
			info.m_valuePtr = m_class->getDeclaredDefault(i);
		}
		virtual void getMemberInfo(int i, hkDataClass::MemberInfo& info) const
		{
			fillMemberInfo( m_class->getMember(i), info );
			info.m_valuePtr = m_class->getDefault(i);
		}
		virtual void getAllMemberInfo(hkArrayBase<hkDataClass::MemberInfo>& infos) const
		{
			HK_ASSERT(0x234324a2, infos.getSize() == getNumMembers());
			hkDataClass::MemberInfo* curMem = infos.end();

			const hkClass* cls = m_class;
			while (cls)
			{
				const int numMem = cls->getNumDeclaredMembers();
				curMem -= numMem;
				for (int i = 0; i < numMem; i++)
				{
					hkDataClass::MemberInfo& info = curMem[i];
					
					fillMemberInfo( cls->getDeclaredMember(i), info );
					info.m_valuePtr = cls->getDeclaredDefault(i);
				}
				// next
				cls = cls->getParent();
			}
			HK_ASSERT(0x4234a234, curMem == infos.begin());
		}
		
		void _getAllMemberHandles(hkArrayBase<hkDataObject::MemberHandle>& handles) const
		{
			HK_ASSERT(0x2342343a, handles.getSize() == getNumMembers());
			hkDataObject::MemberHandle* curHandle = handles.end();

			const hkClass* cls = m_class;
			while (cls)
			{
				const int numMem = cls->getNumDeclaredMembers();
				curHandle -= numMem;
				for (int i = 0; i < numMem; i++)
				{
					curHandle[i] = (hkDataObject::MemberHandle)&cls->getMember(i);
				}
				// next
				cls = cls->getParent();
			}
			HK_ASSERT(0x4234a234, curHandle == handles.begin());
		}

		const hkClass* m_class;
		hkDataWorldNative* m_world;
};

class hkDataArrayNative : public hkDataArrayImpl
{
	public:

		hkDataArrayNative(hkDataWorldNative* world, void* address, int arraySize, int elementStride, hkClassMember::Type type, hkClassMember::Type subType, const hkClass* klass, int tupleCount = 0, void* arrayObject = HK_NULL)
			: m_world(world)
			, m_address(address)
			, m_arraySize(arraySize)
			, m_elementStride(elementStride)
			, m_nativeType(type)
			, m_nativeSubType(subType)
			, m_class(klass)
			, m_tupleCount(tupleCount)
			, m_arrayObject(arrayObject)
		{
			HK_ASSERT(0x26fa9216, m_arraySize >= 0);
			hkDataObject::Type dataType = world->getTypeFromMemberType(type, subType, klass, tupleCount);

			m_dataType = dataType;
			HK_ASSERT(0x34aa99ea, m_elementStride != -1);
		}
		virtual hkDataObject::Type getType() const
		{
			return m_dataType;
		}
		virtual hkDataWorld* getWorld() const
		{
			return m_world;
		}
		virtual void clear()
		{
			HK_ASSERT(0x251e2aeb, m_arrayObject);
			hkVariantDataUtil::clearArray(m_world->m_infoReg, m_arrayObject, m_nativeType, m_class);
		}
		virtual void reserve( int n )
		{
			HK_ASSERT(0x251e2aeb, m_arrayObject);
			m_address = hkVariantDataUtil::reserveArray(m_arrayObject, m_nativeType, m_class, n);
		}
		virtual void setSize(int n)
		{
			if (n != getSize())
			{
				HK_ASSERT(0x251e2aeb, m_arrayObject);
				m_address = hkVariantDataUtil::setArraySize(m_world->m_infoReg, m_arrayObject, m_nativeType, m_class, n);
			}
		}
		virtual int getSize() const
		{
			if (m_arrayObject)
			{
				return hkVariantDataUtil::getArraySize(m_arrayObject);
			}
			else
			{
				return m_arraySize;
			}
		}
		virtual int getUnderlyingIntegerSize() const
		{
			// The underlying size is only used for int arrays
			if(m_nativeType == hkClassMember::TYPE_INT64 || m_nativeType == hkClassMember::TYPE_UINT64)
			{
				return hkSizeOf(hkInt64);
			}
			else
			{
				return hkSizeOf(hkInt32);
			}
		}
		virtual const hkDataClassImpl* getClass() const
		{
			return m_class ? m_world->findClass(m_class->getName()) : HK_NULL;
		}
		virtual const hkReal* asVec(int index) const 
		{
			return hkVariantDataUtil::getReals(m_nativeType, m_tupleCount, hkAddByteOffsetConst(m_address, index*m_elementStride), m_world->m_buffer);
		}
		virtual const char* asString(int index) const 
		{
			return hkVariantDataUtil::getString(m_nativeType, hkAddByteOffsetConst(m_address, index*m_elementStride));
		}
		virtual hkReal asReal(int index) const 
		{
			return hkVariantDataUtil::getReal(m_nativeType, hkAddByteOffsetConst(m_address, index*m_elementStride));
		}
		virtual int asInt(int index) const 
		{
			return (int)hkVariantDataUtil::getInt(m_nativeType, m_nativeSubType, hkAddByteOffsetConst(m_address, index*m_elementStride));
		}
		virtual hkInt64 asInt64(int index) const 
		{
			return hkVariantDataUtil::getInt(m_nativeType, m_nativeSubType, hkAddByteOffsetConst(m_address, index*m_elementStride));
		}
		virtual hkDataObjectImpl* asObject(int index) const 
		{
			hkVariant v = hkVariantDataUtil::getObject(m_nativeType, m_class, m_world->m_vtable, m_world->m_reg, hkAddByteOffsetConst(m_address, index*m_elementStride));
			return (v.m_object && v.m_class) ? hkNativeDataObjectImplCreate(v, m_world) : HK_NULL;
		}

		virtual hkDataArrayImpl* asArray(int index) const
		{
			void* address = hkAddByteOffset(m_address, m_elementStride*index);
			if( m_tupleCount )
			{
				int stride = (m_nativeType == hkClassMember::TYPE_STRUCT) ? m_class->getObjectSize() : hkClassMember::getClassMemberTypeProperties(m_nativeType).m_size;
				HK_ASSERT(0x2d7ae772, stride > 0);
				return new hkDataArrayNative(m_world, address, m_tupleCount, stride, m_nativeType, m_nativeSubType, m_class );
			}
			else
			{
				if (m_nativeType == hkClassMember::TYPE_ARRAY)
				{
					struct Array { void* p; int s; };
					const Array* a = (Array*)address;

					hk_size_t elementSize = hkVariantDataUtil::calcElementSize(m_nativeSubType, hkClassMember::TYPE_VOID, m_class, m_tupleCount);
					HK_ASSERT(0x324a2bc2, elementSize > 0);

					return new hkDataArrayNative(m_world, a->p, a->s, int(elementSize), m_nativeSubType, hkClassMember::TYPE_VOID, m_class);
				}
			}			
			HK_ASSERT(0x43432aab, !"Couldn't create type");
			return HK_NULL;
		}

		virtual hkDataArrayImpl* swizzleObjectMember(const char* name) const
		{
			HK_ASSERT(0x2e6fbb77, m_nativeType == hkClassMember::TYPE_STRUCT);
			const hkClassMember* mem = m_class->getMemberByName(name);
			void* address = hkAddByteOffset(m_address, mem->getOffset());

			hkClassMember::Type mtype = mem->getType();

			return new hkDataArrayNative(m_world, address, m_arraySize, m_elementStride, mtype, mem->getSubType(), mem->getClass(), mem->getCstyleArraySize() );
		}
		hkResult asStridedBasicArray(hkStridedBasicArray& arrOut)
		{
			arrOut.m_tupleSize = 1;
			arrOut.m_type = m_nativeType;

			switch (m_nativeType)
			{
				case hkClassMember::TYPE_FLAGS:
				case hkClassMember::TYPE_ENUM:
				{
					arrOut.m_type = m_nativeSubType;
					arrOut.m_tupleSize = 1;
					break;
				}
				case hkClassMember::TYPE_QUATERNION:
				case hkClassMember::TYPE_VECTOR4:
				{
					arrOut.m_type = hkClassMember::TYPE_REAL;
					arrOut.m_tupleSize = 4;
					break;
				}
				case hkClassMember::TYPE_QSTRANSFORM:
				case hkClassMember::TYPE_ROTATION:
				case hkClassMember::TYPE_MATRIX3:
				{
					arrOut.m_type = hkClassMember::TYPE_REAL;
					arrOut.m_tupleSize = 12;
					break;
				}
				case hkClassMember::TYPE_TRANSFORM:
				case hkClassMember::TYPE_MATRIX4:
				{
					arrOut.m_type = hkClassMember::TYPE_REAL;
					arrOut.m_tupleSize = 16;
					break;
				}
				default: break;
			}

			
			
			// Must get size to handle resizes etc
			arrOut.m_size = getSize();
			arrOut.m_stride = m_elementStride;
			arrOut.m_data = const_cast<void*>(m_address);
			return HK_SUCCESS;
		}

		void _setAll(hkClassMember::Type type, const void* v, int n)
		{
			if (type == hkClassMember::TYPE_UINT8 && m_nativeType == hkClassMember::TYPE_BOOL)
			{
				type = hkClassMember::TYPE_BOOL;
			}
			if (type == hkClassMember::TYPE_UINT8 && m_nativeType == hkClassMember::TYPE_CHAR)
			{
				type = hkClassMember::TYPE_CHAR;
			}

			HK_ASSERT(0x124432aa, type == m_nativeType);
			if (type == m_nativeType)
			{
				setSize(n);
				const int eleSize = hkClassMember::getClassMemberTypeProperties(type).m_size;

				hkString::memCpy(m_address, v, n * eleSize);
			}
		}

		virtual void setAll(const hkBool* v, int n) { _setAll(hkClassMember::TYPE_BOOL, v, n); }
		virtual void setAll(const char* v, int n) { _setAll(hkClassMember::TYPE_CHAR, v, n); }
		virtual void setAll(const hkInt8* v, int n) { _setAll(hkClassMember::TYPE_INT8, v, n); }
		virtual void setAll(const hkUint8* v, int n) { _setAll(hkClassMember::TYPE_UINT8, v, n); }
		virtual void setAll(const hkInt16* v, int n) { _setAll(hkClassMember::TYPE_INT16, v, n); }
		virtual void setAll(const hkUint16* v, int n) { _setAll(hkClassMember::TYPE_UINT16, v, n); }
		virtual void setAll(const hkInt32* v, int n) { _setAll(hkClassMember::TYPE_INT32, v, n); }
		virtual void setAll(const hkUint32* v, int n) { _setAll(hkClassMember::TYPE_UINT32, v, n); }
		virtual void setAll(const hkInt64* v, int n) { _setAll(hkClassMember::TYPE_INT64, v, n); }
		virtual void setAll(const hkUint64* v, int n) { _setAll(hkClassMember::TYPE_UINT64, v, n); }
		virtual void setAll(const hkReal* v, int n) { _setAll(hkClassMember::TYPE_REAL, v, n); }
		virtual void setAll(const hkHalf* v, int n) { _setAll(hkClassMember::TYPE_HALF, v, n); }

		virtual void setVec(int index, const hkReal* val) { hkVariantDataUtil::setReals(m_nativeType, m_tupleCount, val, _getElement(index)); }
		virtual void setString(int index, const char* val) { hkVariantDataUtil::setString(m_nativeType, _getElement(index), val); }
		virtual void setReal(int index, hkReal val) { hkVariantDataUtil::setReal(m_nativeType, _getElement(index), val); }
		virtual void setHalf(int index, hkHalf val) { hkVariantDataUtil::setHalf(m_nativeType, _getElement(index), val); }
		virtual void setInt(int index, int val) { hkVariantDataUtil::setInt(m_nativeType, _getElement(index), val); }
		virtual void setInt64(int index, hkInt64 val) { hkVariantDataUtil::setInt(m_nativeType, _getElement(index), val); }
		virtual void setArray(int index, hkDataArrayImpl* val) { HK_ASSERT(0x7af76c5a, !"Not able to set an array - access through 'asArray'"); }
		virtual void setObject(int index, hkDataObjectImpl* val);


		HK_FORCE_INLINE void* _getElement(int index) { return hkAddByteOffset(m_address, index * m_elementStride); }
		HK_FORCE_INLINE const void* _getElement(int index) const { return hkAddByteOffset(m_address, index * m_elementStride); }


		hkDataWorldNative* m_world;
		void* m_address;
		int m_arraySize;
		int m_elementStride;
		hkClassMember::Type m_nativeType;
		hkClassMember::Type m_nativeSubType;
		const hkClass* m_class;
		hkDataObject::Type m_dataType;
		int m_tupleCount;
		void* m_arrayObject;
};


class hkDataObjectNative : public hkDataObjectImpl
{
	public:

		hkDataObjectNative(const hkVariant& v, hkDataWorldNative* world, hkBool hasOwnedRef = false)
			: m_object( const_cast<hkVariant&>(v) )
			, m_world(world)
			, m_hasOwnedRef(hasOwnedRef)
		{
			HK_ASSERT(0x683aef98, m_world);
			HK_ASSERT(0x438a6ee7, (!v.m_object && !v.m_class) || (v.m_object && v.m_class));
#if defined(HK_DEBUG)
			if( v.m_class )
			{
				const hkDataClassNative* classFromWorld = static_cast<const hkDataClassNative*>(world->findClass(v.m_class->getName()));
				HK_ASSERT(0x777a9c19, classFromWorld && v.m_class == classFromWorld->m_class );
			}
#endif
		}

		hkDataObjectNative(const hkDataObject::Handle& h, hkDataWorldNative* world)
			: m_object( h.p0, h.p1 ? static_cast<hkClass*>(h.p1) : world->m_vtable->getClassFromVirtualInstance(h.p0)  )
			, m_world(world)
			, m_hasOwnedRef(false)
		{
		}

		~hkDataObjectNative()
		{
			if (m_hasOwnedRef)
			{
				HK_ASSERT(0x3424234, m_object.getClass().hasVtable());
				hkReferencedObject* obj = static_cast<hkReferencedObject*>(m_object.getAddress());
				obj->removeReference();
			}
		}

		virtual hkDataObject::Handle getHandle() const
		{
			hkDataObject::Handle h;
			h.p0 = m_object.getAddress();
			const hkClass& k = m_object.getClass();
			// If the class has a vtable, then the raw ptr is enough to uniquely identify it.
			// Indeed, the same object may be accessed with different polymorphic classes.
			// If the object is not virtual, we need the class to differentiate between objects
			// as the first member of a struct. e.g. struct A { }; struct B { A a; }; B b;
			// Then &b == &b.a but they are not the same object.
			h.p1 = k.hasVtable() ? HK_NULL : const_cast<hkClass*>(&k);
			return h;
		}
		virtual const hkDataClassImpl* getClass() const
		{
			const hkClass* c = &m_object.getClass();
			return c ? m_world->findClass(c->getName()) : HK_NULL;
		}
		virtual hkDataObject::Value accessByName(const char* name)
		{
			hkClassMemberAccessor mem = m_object.member(name);
			if (mem.isOk())
			{
				return hkDataObject::Value(this, (MemberHandle)&mem.getClassMember());
			}
			else
			{
				return hkDataObject::Value(HK_NULL, HK_NULL);
			}
		}
		virtual hkBool32 hasMember(const char* name) const
		{
			hkClassMemberAccessor mem = m_object.member(name);
			return mem.isOk();
		}

		virtual hkBool32 isSet(MemberHandle handle)
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			return acc.isOk();
		}
		virtual Iterator getMemberIterator() const
		{
			return 0;
		}
		virtual hkBool32 isValid(Iterator it) const
		{
			return it < m_object.getClass().getNumMembers();
		}
		virtual Iterator getNextMember(Iterator it) const
		{
			return it + 1;
		}
		virtual const char* getMemberName(Iterator it) const
		{
			return m_object.getClass().getMember(it).getName();
		}
		virtual const hkDataObject::Value getMemberValue(Iterator it) const
		{
			const hkClassMember* mem = &m_object.getClass().getMember(it);
			return hkDataObject::Value(const_cast<hkDataObjectNative*>(this), (MemberHandle)mem );
		}

		virtual void getAllMemberHandles(hkArrayBase<hkDataObject::MemberHandle>& handles) const
		{
			const hkDataClassNative* cls = static_cast<const hkDataClassNative*>(getClass());
			HK_ASSERT(0x2342a3a4, cls);
			cls->_getAllMemberHandles(handles);
		}

		virtual void destroy() {}

		virtual hkDataArrayImpl* asArray( MemberHandle handle)
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);

			const hkClass* aclass;
			hkClassMember::Type subType;
			struct Array { void* p; int s; };
			int arraySize;
			void* arrayPtr;
			int elemSize;

			Array* arrPtr = HK_NULL;

			switch( mem->getType() )
			{
				case hkClassMember::TYPE_HOMOGENEOUSARRAY:
				{
					arrPtr = static_cast<Array*>(hkAddByteOffset( acc.getAddress(), sizeof(hkClass*) ));
					const Array& arr = *arrPtr;
					arrayPtr = arr.p;
					arraySize = arr.s;
					aclass = *static_cast<hkClass**>(acc.getAddress());
					if( aclass )
					{
						elemSize = aclass->getObjectSize();
						subType = hkClassMember::TYPE_STRUCT;
					}
					else
					{
						HK_ASSERT(0x571f1bd9, arr.p == HK_NULL && arr.s == 0);
						elemSize = 0;
						subType = hkClassMember::TYPE_VOID;
					}
					break;
				}
				case hkClassMember::TYPE_SIMPLEARRAY:
				case hkClassMember::TYPE_ARRAY:
				{
					arrPtr = static_cast<Array*>(acc.getAddress());
					const Array& arr = *arrPtr;
					arrayPtr = arr.p;
					arraySize = arr.s;
					aclass = mem->getClass();
					subType = mem->getSubType();
					//if( subType == 0 ) return HK_NULL; //hack for nonreflected
					elemSize = hkClassMember::getClassMemberTypeProperties(subType).m_size;
					if( subType == 0 )
					{
						return new hkDataArrayNative( m_world, HK_NULL, 0, 0,hkClassMember::TYPE_VOID, hkClassMember::TYPE_VOID, 0);
					}
					if( elemSize == -1 )
					{
						HK_ASSERT(0x3a225521, subType == hkClassMember::TYPE_STRUCT);
						elemSize = aclass->getObjectSize();
					}
					break;
				}
				case hkClassMember::TYPE_RELARRAY:
				{
					hkRelArray<char> *arr = static_cast<hkRelArray<char>*>(acc.getAddress());
					arrayPtr = arr->begin();
					arraySize = arr->getSize();
					aclass = mem->getClass();
					subType = mem->getSubType();
					//if( subType == 0 ) return HK_NULL; //hack for nonreflected
					elemSize = hkClassMember::getClassMemberTypeProperties(subType).m_size;
					if( subType == 0 )
					{
						return new hkDataArrayNative( m_world, HK_NULL, 0, 0,hkClassMember::TYPE_VOID, hkClassMember::TYPE_VOID, 0);
					}
					if( elemSize == -1 )
					{
						HK_ASSERT(0x3a225521, subType == hkClassMember::TYPE_STRUCT);
						elemSize = aclass->getObjectSize();
					}
					break;
				}
				case hkClassMember::TYPE_BOOL:
				case hkClassMember::TYPE_CHAR:
				case hkClassMember::TYPE_INT8:
				case hkClassMember::TYPE_UINT8:
				case hkClassMember::TYPE_INT16:
				case hkClassMember::TYPE_UINT16:
				case hkClassMember::TYPE_INT32:
				case hkClassMember::TYPE_UINT32:
				case hkClassMember::TYPE_ULONG:
				case hkClassMember::TYPE_INT64:
				case hkClassMember::TYPE_UINT64:
				case hkClassMember::TYPE_REAL:
				case hkClassMember::TYPE_HALF:
				case hkClassMember::TYPE_VECTOR4:
				case hkClassMember::TYPE_QUATERNION:
				case hkClassMember::TYPE_MATRIX3:
				case hkClassMember::TYPE_ROTATION:
				case hkClassMember::TYPE_QSTRANSFORM:
				case hkClassMember::TYPE_MATRIX4:
				case hkClassMember::TYPE_TRANSFORM:
				case hkClassMember::TYPE_STRUCT:
				case hkClassMember::TYPE_POINTER:
				{
					arrayPtr = acc.getAddress();
					arraySize = acc.getClassMember().getCstyleArraySize();
					HK_ASSERT(0x2101cb65, arraySize > 0);
					subType = mem->getType();
					aclass = mem->getClass();
					HK_ASSERT(0x3c14d084, (subType != hkClassMember::TYPE_POINTER && subType != hkClassMember::TYPE_STRUCT) || aclass);
					elemSize = subType == hkClassMember::TYPE_STRUCT ? aclass->getObjectSize() : hkClassMember::getClassMemberTypeProperties(subType).m_size;
					break;
				}
				default:
				{
					HK_ASSERT(0x7a39a4bb, 0);
					arrayPtr = HK_NULL;
					arraySize = HK_NULL;
					aclass = HK_NULL;
					subType = hkClassMember::TYPE_VOID;
					elemSize = 0;
				}
			}

			// The tuple count is zero as you can't have an array of tuples (the tuple has to be wrapped in a class).
			return new hkDataArrayNative( m_world, arrayPtr, arraySize, elemSize, subType, hkClassMember::TYPE_VOID, aclass, 0, arrPtr );
		}
		virtual const char* asString( MemberHandle handle )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			return hkVariantDataUtil::getString(mem->getType(), acc.getAddress());
		}
		virtual hkInt64 asInt( MemberHandle handle )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			return hkVariantDataUtil::getInt(mem->getType(), mem->getSubType(), acc.getAddress());
		}
		virtual hkDataObjectImpl* asObject( MemberHandle handle )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariant v = hkVariantDataUtil::getObject(mem->getType(), mem->getClass(), m_world->m_vtable, m_world->m_reg, acc.asRaw());
			return (v.m_object && v.m_class) ? hkNativeDataObjectImplCreate( v, m_world ) : HK_NULL;
		}
		virtual const hkReal* asVec( MemberHandle handle, int nreal )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			return hkVariantDataUtil::getReals(mem->getType(), mem->getCstyleArraySize(), acc.getAddress(), m_world->m_buffer);
		}
		virtual hkReal asReal( MemberHandle handle )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			return hkVariantDataUtil::getReal(mem->getType(), acc.getAddress());
		}
		virtual void assign( MemberHandle handle, const hkDataObject::Value& valueIn )
		{
			assignValueImpl( this, handle, valueIn);
		}
		virtual void assign( MemberHandle handle, hkDataObjectImpl* val )
		{
			const hkClassMember* clsMem = (const hkClassMember*)handle;

			// Get the class member and dest
			//const hkClass& dstCls = m_object.getClass();
			
			HK_ASSERT(0x4323a2a3, clsMem);
			// Can't set a struct this way. Only way to access a struct is to do 'asObject' and modify it.
			HK_ASSERT(0x4324a324, clsMem->getType() != hkClassMember::TYPE_STRUCT);

			const hkBool isRef = !clsMem->getFlags().allAreSet(hkClassMember::NOT_OWNED);

			hkClassMemberAccessor mem = m_object.member(clsMem);	
			if (val)
			{
				hkDataObjectNative* nativeObj = static_cast<hkDataObjectNative*>(val);
				HK_ASSERT(0x234ab24a, nativeObj);
				hkVariantDataUtil::setPointer(nativeObj->m_object.getClass(), nativeObj->m_object.getAddress(), (void**)mem.getAddress(), isRef);
			}
			else
			{
				if (clsMem->getClass())
				{
					hkVariantDataUtil::setPointer(*clsMem->getClass(), HK_NULL, (void**)mem.getAddress(), isRef);
				}
			}
		}
		virtual void assign( MemberHandle handle, hkDataArrayImpl* value )
		{
			HK_ASSERT2(0x793b0264, false, "Cannot assign - but can access and modify with asArray().");
		}
		virtual void assign( MemberHandle handle, const char* value )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariantDataUtil::setString(mem->getType(), acc.getAddress(), value);
		}
		virtual void assign( MemberHandle handle, hkReal r )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariantDataUtil::setReal(mem->getType(), acc.getAddress(), r);
		}
		virtual void assign( MemberHandle handle, hkHalf r )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariantDataUtil::setReal(mem->getType(), acc.getAddress(), r);
		}
		virtual void assign( MemberHandle handle, const hkReal* r, int nreal )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariantDataUtil::setReals(mem->getType(), mem->getCstyleArraySize(), r, acc.getAddress(), nreal);
		}
		virtual void assign( MemberHandle handle, hkInt64 valueIn )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariantDataUtil::setInt(mem->getType(), mem->getSubType(), acc.getAddress(), valueIn);
		}
		virtual void assign( MemberHandle handle, int valueIn )
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkClassMemberAccessor acc = m_object.member(mem);
			hkVariantDataUtil::setInt(mem->getType(), mem->getSubType(), acc.getAddress(), valueIn);
		}
		virtual void getMemberInfo( MemberHandle handle, hkDataClass::MemberInfo& infoOut)
		{
			const hkClassMember* mem = (const hkClassMember*)handle;
			hkDataClassNative::_fillMemberInfo(HK_NULL, m_world, *mem, infoOut);
		}

		virtual hkBool32 createdDuringPatching() const
		{
			return false;
		}

		hkClassAccessor m_object;
		hkDataWorldNative* m_world;
		hkBool m_hasOwnedRef;				///< if set the object must be hkReferencedObject derived and its owned by this wrapper.
};

hkDataObjectImpl* hkNativeDataObjectImplCreate(hkVariant& v, hkDataWorldNative* w, hkBool isRefOwned)
{
	if( v.m_class )
	{
		if( const hkDataClassNative* k = static_cast<const hkDataClassNative*>(w->findClass(v.m_class->getName())) )
		{
			v.m_class = k->m_class;
		}
		else
		{
			HK_WARN_ALWAYS(0x1800473a, "Ignore object at 0x" << v.m_object << ". Class '" << v.m_class->getName() << "' is not registered in the provided hkDataWorldNative.");
			v.m_object = HK_NULL;
			v.m_class = HK_NULL;
		}
	}
	return new hkDataObjectNative(v, w, isRefOwned);
}


//////////////////////////////////////////////////////////////////////////
// Data World Native
//////////////////////////////////////////////////////////////////////////



void hkDataArrayNative::setObject(int index, hkDataObjectImpl* val) 
{
	// Check it belongs to the right world
	HK_ASSERT(0x3242432a, index >= 0 && index < getSize());

	hkDataObjectNative* nativeObj = static_cast<hkDataObjectNative*>(val);
	HK_ASSERT(0x234ab24a, nativeObj);

	void* dstAddr = hkAddByteOffset(m_address, m_elementStride * index);

	if (m_nativeType == hkClassMember::TYPE_STRUCT)
	{
		HK_ASSERT(0x243432, !"Unable to set a struct - access as 'asObject' and write members");
	}
	else
	{
		// set it	
		hkVariantDataUtil::setPointer(nativeObj->m_object.getClass(), nativeObj->m_object.getAddress(), (void**)dstAddr);
	}
}

//////////////////////////////////////////////////////////////////////////
// Data World Native
//////////////////////////////////////////////////////////////////////////

hkDataWorldNative::hkDataWorldNative(hkBool accessSerializeIgnored)
: m_accessSerializeIgnored(accessSerializeIgnored)
{
	m_contents.m_class = HK_NULL;
	m_contents.m_object = HK_NULL;
	m_vtable = hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry();
	m_reg = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
	m_infoReg = hkBuiltinTypeRegistry::getInstance().getTypeInfoRegistry();
}

hkDataWorldNative::~hkDataWorldNative()
{
	typedef hkStringMap<hkDataClassNative*> Map;
	for( Map::Iterator it = m_classes.getIterator(); m_classes.isValid(it); it = m_classes.getNext(it) )
	{
		delete m_classes.getValue(it);
	}
}

hkDataObject::Type hkDataWorldNative::getTypeFromMemberType(hkClassMember::Type mtype, hkClassMember::Type stype, const hkClass* klass, int count)
{
	return hkDataObjectUtil::getTypeFromMemberType(m_typeManager, mtype, stype, klass, count);
}

void hkDataWorldNative::setClassRegistry(const hkClassNameRegistry* r)
{
	m_reg = r;
}

void hkDataWorldNative::setVtableRegistry(const hkVtableClassRegistry* vt)
{
	m_vtable = vt;
}

hkDataObjectImpl* hkDataWorldNative::wrapObject(void* object, const hkClass& klass) const
{
	HK_ASSERT(0x225571c0, object);
	hkVariant v = { object, &klass };
	return hkNativeDataObjectImplCreate(v, const_cast<hkDataWorldNative*>(this));
}

void hkDataWorldNative::setContents(void* object, const hkClass& klass)
{
	HK_ASSERT(0x1d5f821a, m_vtable && m_reg);
	m_contents.m_object = object;
	
	const hkClass* registeredClass = m_reg->getClassByName(klass.getName());
	HK_ASSERT3(0x1800473b, registeredClass, "Cannot set contents with object at 0x" << object << ". Class '" << klass.getName() << "' is not registered in the hkDataWorldNative. If this is a Havok class, make sure the class's product reflection is enabled near where hkProductFeatures.cxx is included. Otherwise, check your own class registration.");

	if( klass.hasVtable() )
	{
		const hkClass* mostDerived = hkVariantDataUtil::findMostDerivedClass(object, m_vtable, m_reg);
		HK_ASSERT3(0x2e386252, mostDerived, "Cannot find class for object at 0x" << object << ". One possible mistake is passing a pointer to a pointer, instead of a pointer." );
		HK_ASSERT3(0x56db8a19, registeredClass->isSuperClass(*mostDerived), "Mismatch between object and specified class type: " << klass.getName() << " isn't a parent of " << mostDerived->getName() << "." );
		m_contents.m_class = mostDerived;
	}
	else
	{
		m_contents.m_class = registeredClass;
	}

	HK_ASSERT3(0x1800473b, m_contents.m_class, "Cannot set contents with object at 0x" << object << ". Class '" << klass.getName() << "' is not registered in the hkDataWorldNative. If this is a Havok class, make sure the class's product reflection is enabled near where hkProductFeatures.cxx is included. Otherwise, check your own class registration.");

	if( m_contents.m_class == HK_NULL )
	{
		m_contents.m_object = HK_NULL;
	}
}

hkDataClassImpl* hkDataWorldNative::findClass(const char* name) const
{
	if( hkDataClassImpl* ci = ( name ? m_classes.getWithDefault(name, HK_NULL) : HK_NULL ) )
	{
		return ci;
	}
	const hkClass* k = name ? m_reg->getClassByName(name) : HK_NULL;
	HK_ASSERT3( 0xf0df3423, k, "Cannot find class with name: "<<name);
	hkDataClassNative* ci = new hkDataClassNative(k, const_cast<hkDataWorldNative*>(this));
	m_classes.insert(name,ci);
	return ci;
}

void hkDataWorldNative::findAllClasses(hkArray<hkDataClassImpl*>::Temp& classesOut) const
{
	if( !m_reg )
	{
		return;
	}

	hkArray<const hkClass*> classes;
	m_reg->getClasses(classes);
	for( int i = 0; i < classes.getSize(); ++i )
	{
		if( !classes[i]->getFlags().allAreSet(hkClass::FLAGS_NOT_SERIALIZABLE) )
		{
			hkDataClassImpl* c = findClass(classes[i]->getName());
			HK_ASSERT(0x4e227ff7, c);
			classesOut.pushBack(c);
		}
	}
}

hkDataObject hkDataWorldNative::findObject(const hkDataObject::Handle& handle) const
{
	return new hkDataObjectNative(handle, const_cast<hkDataWorldNative*>(this));
}

hkEnum<hkDataWorld::DataWorldType, hkInt32> hkDataWorldNative::getType() const
{
	return TYPE_NATIVE;
}

hkDataObject hkDataWorldNative::getContents() const
{
	return new hkDataObjectNative(m_contents, const_cast<hkDataWorldNative*>(this));
}

hkDataObjectImpl* hkDataWorldNative::newObject(const hkDataClass& klass, bool) const
{
	HK_ASSERT(0x10ce8c0e, m_reg);
	const hkClass* k = m_reg->getClassByName( klass.getName() );
	HK_ASSERT(0x2699fdd2, k);
	hkVariant v;
	v.m_object = hkAllocateChunk<char>( k->getObjectSize(), HK_MEMORY_CLASS_SERIALIZE);
	v.m_class = k;
	if( v.m_object )
	{
		hkString::memSet(v.m_object, 0, k->getObjectSize());
		if( const hkTypeInfo* t = hkBuiltinTypeRegistry::getInstance().getLoadedObjectRegistry()->getTypeInfo( klass.getName() ) )
		{
			t->finishLoadedObject(v.m_object, 0);
		}
	}
	return hkNativeDataObjectImplCreate(v, const_cast<hkDataWorldNative*>(this));
}

hkDataClassImpl* hkDataWorldNative::newClass(const hkDataClass::Cinfo& cinfo)
{
	return HK_NULL;
}

hkDataArrayImpl* hkDataWorldNative::newArray(hkDataObject& _obj, hkDataObject::MemberHandle handle, const hkDataClass::MemberInfo& minfo) const
{
	HK_ASSERT(0x21724e3f, 0);
	// 	hkDataObjectNative* obj = static_cast<hkDataObjectNative*>( _obj.getImplementation() );
	// 	const hkClass* klass = m_reg->getClassByName( minfo.m_owner->getName() );
	// 	return new hkDataArrayNative( this, obj->m_object, 0, -1, -1, mmm );
	return HK_NULL;
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
