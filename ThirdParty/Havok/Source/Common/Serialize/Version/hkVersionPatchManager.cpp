/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Serialize/hkSerialize.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Reflection/Registry/hkClassNameRegistry.h>
#include <Common/Serialize/Data/Dict/hkDataObjectDict.h>
#include <Common/Serialize/Data/hkDataObjectImpl.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/Config/hkConfigVersion.h>

#if 0
#	include <Common/Base/Fwd/hkcstdio.h>
#	include <Common/Base/DebugUtil/Logger/hkOstreamLogger.h>
	static hkLogger* _logger;
#	define LOG(A) if( _logger ) _logger->debug A
#	define TRACE(A) printf A
#else
#	define LOG(A)
#	define TRACE(A)
#endif

HK_SINGLETON_IMPLEMENTATION(hkVersionPatchManager);

namespace
{
	struct ClassVersion
	{
		const char* name;
		int version;
		hkBool32 operator==(const ClassVersion& v) { return version == v.version && hkString::strCmp(name,v.name)==0; }
	};
}

hkDefaultClassWrapper::hkDefaultClassWrapper(const hkClassNameRegistry* nameReg)
{
	if (!nameReg)
	{
		nameReg = hkBuiltinTypeRegistry::getInstance().getClassNameRegistry();
	}
	HK_ASSERT(0x324324b4, nameReg);

	m_nameReg = nameReg;
}

hkDataClassImpl* hkDefaultClassWrapper::wrapClass(hkDataWorld* world, const char* typeName)
{
	hkDataClassImpl* clsImpl = world->findClass(typeName);
	if (clsImpl)
	{
		return clsImpl;
	}

	const hkClass* k = m_nameReg->getClassByName(typeName); // present
	if (!k)
	{
		return HK_NULL;
	}

	if (world->getType() == hkDataWorld::TYPE_DICTIONARY)
	{
		hkDataWorldDict* dictWorld = static_cast<hkDataWorldDict*>(world);

		return dictWorld->wrapClass(*k);
	}

	return HK_NULL;
}

static const char* reportPendingDependencies(const hkArray<int>& pending, const hkArray<const hkVersionPatchManager::PatchInfo*>& patchInfos, hkStringBuf& dependencies)
{
#define NAME(IDX) (patchInfos[(IDX)]->oldName ? patchInfos[(IDX)]->oldName : patchInfos[(IDX)]->newName)
#define VERS(IDX) (patchInfos[(IDX)]->oldName ? patchInfos[(IDX)]->oldVersion : patchInfos[(IDX)]->newVersion)

	dependencies.clear();
	for( int i = 0; i < pending.getSize(); ++i )
	{
		dependencies.appendPrintf("%s(%x)...", NAME(pending[i]), VERS(pending[i]));
	}
	return dependencies;
#undef NAME
#undef VERS
}

static int walkDependencies( int curIndex, hkArray<int>& order, const hkSerializeMultiMap<int,int>& incoming, int counter, hkArray<int>& pending, const hkArray<const hkVersionPatchManager::PatchInfo*>& patchInfos )
{
	enum Status { UNSEEN = -1, PENDING = -2 /*DONE >= 0 */ };

	if( order[curIndex] == UNSEEN )
	{
		order[curIndex] = PENDING;
		pending.pushBack(curIndex);
		for( int it = incoming.getFirstIndex( curIndex );
			it != -1;
			it = incoming.getNextIndex(it) )
		{
			int precedingIndex = incoming.getValue(it);
			switch( order[precedingIndex] )
			{
				case UNSEEN:
					counter = walkDependencies( precedingIndex, order, incoming, counter, pending, patchInfos );
					break;
				case PENDING:
				{
					pending.pushBack(precedingIndex);
					hkStringBuf deps;
					HK_ASSERT3(0x6171e278, false, "Circular: " << reportPendingDependencies(pending, patchInfos, deps));
					pending.popBack();
					break;
				}
				default:
					break;
			}
		}
		pending.popBack();
		order[curIndex] = counter++;
	}
	return counter;
}

hkVersionPatchManager::UidFromClassVersion::~UidFromClassVersion()
{
	for( hkStringMap<const char*>::Iterator it = m_cachedNames.getIterator(); m_cachedNames.isValid(it); it = m_cachedNames.getNext(it) )
	{
		hkDeallocate<char>(m_cachedNames.getValue(it));
	}
}

hkUint64 hkVersionPatchManager::UidFromClassVersion::get( const char* name, int ver ) const
{
	HK_ASSERT(0x345e3567, m_indexFromName.getSize() == m_names.getSize() );
	const char* cachedName = cache(name);
	int nid = m_indexFromName.getOrInsert( cachedName, m_names.getSize() );
	if( nid == m_names.getSize() )
	{
		m_names.pushBack(cachedName);
	}
	return (hkUint64(ver) << 32) | unsigned(nid);
}

const char* hkVersionPatchManager::UidFromClassVersion::getName( hkUint64 uid ) const
{
	int idx = int(uid);
	return m_names[idx];
}

int hkVersionPatchManager::UidFromClassVersion::getVersion( hkUint64 uid ) const
{
	return int(uid>>32);
}

const char* hkVersionPatchManager::UidFromClassVersion::cache(const char* name) const
{
	if( name )
	{
		char* cached = m_cachedNames.getWithDefault(name, HK_NULL);
		if( cached == HK_NULL )
		{
			cached = hkString::strDup(name);
			m_cachedNames.insert(cached, cached);
		}
		return cached;
	}
	return HK_NULL;
}

hkVersionPatchManager::hkVersionPatchManager()
{
	m_uidFromClassVersion = new UidFromClassVersion();
}

void hkVersionPatchManager::clearPatches()
{
	m_patchInfos.clear();
	m_patchIndexFromUid.clear();

	delete m_uidFromClassVersion;
	m_uidFromClassVersion = new UidFromClassVersion();
}

//
//	Clears all patches belonging to the given product. The product match is done by name prefix.

void hkVersionPatchManager::clearProductPatches(const char* productPrefix)
{
	const int prefixLen = hkString::strLen(productPrefix);

	for (int k = m_patchInfos.getSize() - 1; k >= 0; k--)
	{
		const PatchInfo* p = m_patchInfos[k];
		if ( p->newName && !hkString::strNcmp(productPrefix, p->newName, prefixLen) )
		{
			m_patchInfos.removeAt(k);
		}
	}
	m_patchIndexFromUid.clear();

	delete m_uidFromClassVersion;
	m_uidFromClassVersion = new UidFromClassVersion();
}

hkVersionPatchManager::~hkVersionPatchManager()
{
	delete m_uidFromClassVersion;
}

const hkArray<const hkVersionPatchManager::PatchInfo*>& hkVersionPatchManager::getPatches() const
{
	return m_patchInfos;
}

hkUint64 hkVersionPatchManager::getUid( const char* name, int ver ) const
{
	return m_uidFromClassVersion->get(name, ver);
}

const char* hkVersionPatchManager::getClassName( hkUint64 uid ) const
{
	return m_uidFromClassVersion->getName(uid);
}

hkInt32 hkVersionPatchManager::getClassVersion( hkUint64 uid ) const
{
	return m_uidFromClassVersion->getVersion(uid);
}

hkBool32 isValidPatch(const hkVersionPatchManager::PatchInfo* patch)
{
	if( patch->oldVersion == -1/*HK_CLASS_ADDED*/ )
	{
		if( patch->oldName != HK_NULL )
		{
			HK_ASSERT3(0x21be641e, false, "Found incorrectly defined patch. Add new class patch (HK_CLASS_ADDED) has 'source' class name  set to '" << patch->oldName << "', but must be HK_NULL.");
			return false;
		}
		if( patch->newVersion == -1/*HK_CLASS_REMOVED*/ )
		{
			HK_ASSERT2(0x66920794, false, "Found incorrectly defined patch. Add new class patch (HK_CLASS_ADDED) has 'destination' class version set to HK_CLASS_REMOVED.");
			return false;
		}
		if( patch->newName == HK_NULL )
		{
			HK_ASSERT2(0x60dc0f50, false, "Found incorrectly defined patch. Add new class patch (HK_CLASS_ADDED) has 'destination' class name set to null.");
			return false;
		}
	}
	else if( patch->newVersion == -1/*HK_CLASS_REMOVED*/ )
	{
		if( patch->oldName == HK_NULL )
		{
			HK_ASSERT2(0x7178f8cf, false, "Found incorrectly defined patch. Remove class patch (HK_CLASS_REMOVED) has 'source' class name set to null.");
			return false;
		}
		if( patch->newName != HK_NULL )
		{
			HK_ASSERT3(0x69e2d0b3, false, "Found incorrectly defined patch. Remove class patch (HK_CLASS_REMOVED) has 'destination' class name  set to '" << patch->newName << "', but must be HK_NULL.");
			return false;
		}
	}
	else
	{
		if( patch->oldName == HK_NULL )
		{
			HK_ASSERT2(0x42f53922, false, "Found incorrectly defined patch. Update class patch has 'source' class name set to HK_NULL.");
			return false;
		}
	}
	return true;
}

int hkVersionPatchManager::findLastPatchIndexForUid(hkUint64 uid, hkBool32 allowRenames) const
{
	for(int index = m_patchIndexFromUid.getWithDefault(uid, -1);
		index != -1;
		index = m_patchIndexFromUid.getWithDefault(uid, -1) )
	{
		const hkVersionPatchManager::PatchInfo* p = m_patchInfos[index];
		HK_ASSERT(0x7f2cd63e, p && p->oldName);
		if( p->newVersion == -1/*HK_CLASS_REMOVED*/
			|| (!allowRenames && p->newName && hkString::strCmp(p->oldName, p->newName) != 0) )
		{
			return index;
		}
		uid = m_uidFromClassVersion->get(p->newName, p->newVersion);
	}
	return -1;
}

static hkBool32 findPatchIndexInDependencies(int indexToFind, int startIndex, const hkSerializeMultiMap<int, int>& incoming, hkPointerMap<int, int>& done)
{
	if( done.hasKey(startIndex) == false )
	{
		done.insert(startIndex, 0);
		for( int it = incoming.getFirstIndex(startIndex); it != -1; it = incoming.getNextIndex(it) )
		{
			int foundIndex = incoming.getValue(it);
			if( indexToFind == foundIndex )
			{
				return true;
			}
			if( findPatchIndexInDependencies(indexToFind, foundIndex, incoming, done) )
			{
				return true;
			}
		}
	}
	return false;
}

hkResult hkVersionPatchManager::recomputePatchDependencies() const
{
	// todo read/write locks
	hkPointerMap<hkUint64, hkInt32> idxNewSrcFromUid;
	const hkPointerMap<hkUint64, int>& idxSrcFromUid = m_patchIndexFromUid;
	hkSerializeMultiMap<hkUint64, hkInt32> idxDstFromUid;
	const hkVersionPatchManager::UidFromClassVersion* uidFromTuple = m_uidFromClassVersion;
	hkSerializeMultiMap<const char*, hkInt32> lastClassPatchIndexesFromClassName;
	{
		m_patchIndexFromUid.clear();
		for( int i = 0; i < m_patchInfos.getSize(); ++i )
		{
			const PatchInfo* p = m_patchInfos[i];
			if( !isValidPatch(p) )
			{
				return HK_FAILURE;
			}
			if( p->oldVersion != -1 /*HK_CLASS_ADDED*/)
			{
				hkUint64 uid = m_uidFromClassVersion->get(p->oldName, p->oldVersion);
				if( m_patchIndexFromUid.hasKey(uid) == false )
				{
					m_patchIndexFromUid.insert( uid, i );
				}
				else
				{
					HK_ASSERT3(0x6a12698e, false, "Found duplicated patch for class '" << p->oldName << "' (version " << p->oldVersion << ").\n"\
						"Make sure you register the patch once with hkVersionPatchManager.");
					return HK_FAILURE;
				}
				if( p->newVersion == -1 /*HK_CLASS_REMOVED*/ || (p->newName && hkString::strCmp(p->oldName, p->newName) != 0 /*renamed*/) )
				{
					lastClassPatchIndexesFromClassName.insert(p->oldName, i); // last known patch index for p->oldName class
				}
			}
			else
			{
				hkUint64 uid = m_uidFromClassVersion->get(p->newName, p->newVersion);
				if( idxNewSrcFromUid.hasKey(uid) == false )
				{
					idxNewSrcFromUid.insert( uid, i );
				}
				else
				{
					HK_ASSERT3(0x6b12698f, false, "Found duplicated patch for new class '" << p->newName << "' (version " << p->newVersion << ").\n"\
						"Make sure you register the new class patch once with hkVersionPatchManager.");
					return HK_FAILURE;
				}
			}
			if( p->newVersion != -1/*HK_CLASS_REMOVED*/ )
			{
				hkUint64 newUid = uidFromTuple->get(p->newName ? p->newName : p->oldName, p->newVersion);
				idxDstFromUid.insert(newUid, i);
			}
		}
	}

	TRACE(("There are %i patches to consider", m_patchInfos.getSize() - 1 ));
	hkStringMap<int> patchIndexFromClassName; // used to check conflicts among patch groups with the same class name
	hkSerializeMultiMap<int, int> incoming; // patch index key comes after patch index value
	hkSerializeMultiMap<int, int> multiIncoming; // patch index key comes after patch index value for patch groups with the same class name
 	for( int patchIndex = 0; patchIndex < m_patchInfos.getSize(); ++patchIndex )
 	{
		const PatchInfo* pinfo = m_patchInfos[patchIndex];
		const char* curName = pinfo->oldName;
		int curVersion = pinfo->oldVersion;
		if( pinfo->oldVersion == -1 /*HK_CLASS_ADDED*/)
		{
			// new class patch (may have dependencies)
			curName = pinfo->newName;
			curVersion = pinfo->newVersion;
			if( !curName || curVersion == -1 /*HK_CLASS_REMOVED*/ )
			{
				HK_ASSERT2(0x36b2fc1e, false, "Found incorrectly defined patch. Class cannot be added and removed using one patch.");
				return HK_FAILURE;
			}
		}
		hkUint64 curUid = uidFromTuple->get( curName, curVersion );
		int curIdx;
		if( pinfo->oldVersion == -1 /*HK_CLASS_ADDED*/)
		{
			curIdx = idxNewSrcFromUid.getWithDefault(curUid, -1);
			HK_ASSERT(0x14f5428f, curIdx == patchIndex);
		}
		else
		{
			curIdx = idxSrcFromUid.getWithDefault( curUid, -1 );
			// update class patch
			HK_ASSERT(0x14f5427f, curIdx == patchIndex);
#if defined(TRACE)
#	define NAME(IDX) (m_patchInfos[(IDX)]->oldName ? m_patchInfos[(IDX)]->oldName : m_patchInfos[(IDX)]->newName)
#	define VERS(IDX) (m_patchInfos[(IDX)]->oldName ? m_patchInfos[(IDX)]->oldVersion : m_patchInfos[(IDX)]->newVersion)
#endif
			for( int it = idxDstFromUid.getFirstIndex( curUid );
				it != -1;
				it = idxDstFromUid.getNextIndex(it) )
			{
				int dstIdx = idxDstFromUid.getValue(it);
				incoming.insert( curIdx, dstIdx ); // patch producing me comes first
				//TRACE(("\t%s_%x -> %s_%x [color=\"direct\" constraint=false]\n", NAME(dstIdx), VERS(dstIdx), NAME(curIdx), VERS(curIdx) ));
				TRACE(("\t%d (%d)\n", curIdx, dstIdx));
			}
		}

		// check dependecies
		for( int ci = 0; ci < pinfo->numComponent; ++ci )
		{
			const PatchInfo::Component& pc = pinfo->component[ci];
			if( pc.type == PATCH_DEPENDS )
			{
				const DependsPatch* dp = static_cast<const DependsPatch*>( pc.patch);
				hkUint64 dpUid = uidFromTuple->get( dp->name, dp->version );
				{
					// patch modifying my dependency must come after me
					int dpIdx = idxSrcFromUid.getWithDefault( dpUid, -1 );
					if( dpIdx != -1 )
					{
						incoming.insert( dpIdx, curIdx ); // I must come before patch for my dependency
						//TRACE(("\t%s_%x -> %s_%x [label=\"dep comes after\"]\n", NAME(curIdx), VERS(curIdx), NAME(dpIdx), VERS(dpIdx) ));
						TRACE(("\t%d (%d)\n", dpIdx, curIdx));
					}
				}

				// I must come after all patches producing my dependency
				for( int it = idxDstFromUid.getFirstIndex( dpUid );
					it != -1;
					it = idxDstFromUid.getNextIndex(it) )
				{
					int dpOutputIdx = idxDstFromUid.getValue(it);
					incoming.insert( curIdx, dpOutputIdx ); // I come after patches producing my dependency
					//TRACE(("\t%s_%x -> %s_%x [color=\"he helps before\" constraint=false]\n", NAME(dpOutputIdx), VERS(dpOutputIdx), NAME(curIdx), VERS(curIdx) ));
					TRACE(("\t%d (%d)\n", curIdx, dpOutputIdx));
				}
			}
		}
		// check groups
		if( pinfo->oldVersion == -1 /*HK_CLASS_ADDED*/ && lastClassPatchIndexesFromClassName.getFirstIndex(curName) != -1 )
		{
			int it = lastClassPatchIndexesFromClassName.getFirstIndex(curName);
			if( it != -1 )
			{
				int curLastIdx = findLastPatchIndexForUid(curUid);
				if( curLastIdx == -1 ) // -1 = class still exists, so this patch must be after all other group patches with the same class name
				{
					int lastPatchIndex = patchIndexFromClassName.getWithDefault(curName, -1);
					if( lastPatchIndex != -1 )
					{
						HK_ASSERT3(0x4a21c0fc, false, "Found conflict in registered patches for class '" << curName << "'.\n"
							"The class patch for version " << m_patchInfos[lastPatchIndex]->newVersion << " conflicts with patch for new class version " << curVersion << ".");
						return HK_FAILURE;
					}
					patchIndexFromClassName.insert(curName, patchIndex);
				}
				hkSerializeMultiMap<int, int>& dependencyMap = curLastIdx == -1 ? incoming : multiIncoming;
				for( ; it != -1; it = lastClassPatchIndexesFromClassName.getNextIndex(it) )
				{
					int prevIdx = lastClassPatchIndexesFromClassName.getValue(it);
					if( curLastIdx != prevIdx )
					{
						dependencyMap.insert( curIdx, prevIdx ); // I come after all patches for previously removed class
						if( curLastIdx == -1 )
						{
							TRACE((".\t%d (%d)\n", curIdx, prevIdx));
						}
					}
				}
			}
		}
	}
	// update 'incoming' with patch group dependencies
	for( hkPointerMap<int, int>::Iterator it = multiIncoming.m_indexMap.getIterator(); multiIncoming.m_indexMap.isValid(it); it = multiIncoming.m_indexMap.getNext(it) )
	{
		int beginPatchIndex = multiIncoming.m_indexMap.getKey(it);
		for( int mi = multiIncoming.getFirstIndex(beginPatchIndex); mi != -1; mi = multiIncoming.getNextIndex(mi) )
		{
			hkPointerMap<int, int> done;
			// make sure there is no circular dependency walking backward
			int startIndex = multiIncoming.getValue(mi);
			if( !findPatchIndexInDependencies(beginPatchIndex, startIndex, incoming, done) )
			{
				// add dependency
				incoming.insert(beginPatchIndex, startIndex);
				TRACE(("+\t%d (%d)\n", beginPatchIndex, startIndex));
			}
		}
	}
	TRACE(("}\n"));


	hkArray<const PatchInfo*> allPatchInfos; allPatchInfos.swap(m_patchInfos);
	// 0=< done
	// -1 pending
	// -2 done
	hkArray<int> order; order.setSize( allPatchInfos.getSize(), -1 );

	hkArray<int> pending;
	int counter = 0;
	for( int patchIndex = 0; patchIndex < allPatchInfos.getSize(); ++patchIndex )
	{
		counter = walkDependencies( patchIndex, order, incoming, counter, pending, allPatchInfos );
	}
	m_patchInfos.setSize(allPatchInfos.getSize());
	for( int patchIndex = 0; patchIndex < allPatchInfos.getSize(); ++patchIndex )
	{
		int correctIdx = order[patchIndex];
		if( correctIdx < 0 )
		{
			HK_ASSERT2(0x3fecb5e3, false, "Found circular dependency in patches.");
			m_patchInfos.swap(allPatchInfos);
			return HK_FAILURE;
		}
		m_patchInfos[correctIdx] = allPatchInfos[patchIndex];
	}

	{
		m_patchIndexFromUid.clear();
		for( int i = 0; i < m_patchInfos.getSize(); ++i )
		{
			const PatchInfo* p = m_patchInfos[i];
			if( p->oldName )
			{
				hkUint64 uid = m_uidFromClassVersion->get( p->oldName, p->oldVersion);
				HK_ASSERT(0x6c12698e, m_patchIndexFromUid.hasKey(uid) == false );
				m_patchIndexFromUid.insert( uid, i );
			}
			else
			{
				HK_ASSERT(0x3754c508, p->newName);
			}
		}
	}
	return HK_SUCCESS;
}

template <typename Elem>
struct HeapArray
{
	void insert( const Elem& e )
	{
		for( int i = 0; i < m_elems.getSize(); ++i )
		{
			if( e < m_elems[i] )
			{
				m_elems.insertAt(i, e);
				return;
			}
		}
		m_elems.pushBack(e);
	}
	const Elem& top() const
	{
		return m_elems[0];
	}
	int getSize() const
	{
		return m_elems.getSize();
	}
	void replaceTop( const Elem& e )
	{
		popTop();
		insert(e);
	}
	void popTop()
	{
		m_elems.removeAtAndCopy(0);
	}
	bool hasElement( const Elem& e )
	{
		for( int i = 0; i < m_elems.getSize(); ++i )
		{
			if( e == m_elems[i] )
			{
				return true;
			}
		}
		return false;
	}
	hkArray<Elem> m_elems;
};

struct UidItem
{
	hkUint64 uid; int patchIndex;
	bool operator<(UidItem p) const { return patchIndex < p.patchIndex; }
	bool operator==(UidItem p) const { return uid == p.uid; }
};

static void setAllDependeciesToDo(HeapArray<UidItem>& todo, const char* classname, int version,
	const hkDataWorld& worldToUpdate, const hkVersionPatchManager& man, const hkPointerMap<hkUint64, hkInt32>& idxNewSrcFromUid,
	const hkPointerMap<hkUint64, hkInt32>& patchesDone)
{
	hkUint64 uid = man.getUid( classname, version );
	hkDataClassImpl* k = worldToUpdate.findClass(classname);
	int patchIndex = -1;
	if( !k || k->getVersion() != version ) // find 'HK_CLASS_ADDED' patch first
	{
		patchIndex = idxNewSrcFromUid.getWithDefault(uid, -1);
	}
	/* 0 - NONE, 1 - 'HK_CLASS_ADDED' patch, 2 - update patch */
	hkInt32 patchDoneFlag = patchesDone.getWithDefault(uid, 0);
	if( patchIndex == -1 || patchDoneFlag == 1 ) // uid is not 'HK_CLASS_ADDED' patch, or the uid for 'HK_CLASS_PATCH' is already processed
	{
		patchIndex = man.getPatchIndex(uid);
	}
	UidItem p = { uid, patchIndex };
	if( patchDoneFlag != 2 && !todo.hasElement( p ) )
	{
		todo.insert(p);
		TRACE(("++\tclass '%s' ver %d\n", classname, version));
	}
	if( patchIndex < 0 )
	{
		return;
	}
	const hkVersionPatchManager::PatchInfo* pinfo = man.getPatch(patchIndex);
	for( int componentIndex = pinfo->numComponent - 1; componentIndex >= 0 ; --componentIndex )
	{
		const hkVersionPatchManager::PatchInfo::Component& pc = pinfo->component[componentIndex];
		switch( pc.type )
		{
			case hkVersionPatchManager::PATCH_DEPENDS:
			{
				const hkVersionPatchManager::DependsPatch* dp = reinterpret_cast<const hkVersionPatchManager::DependsPatch*>(pc.patch);
				setAllDependeciesToDo(todo, dp->name, dp->version, worldToUpdate, man, idxNewSrcFromUid, patchesDone);
				break;
			}
			default:
			{
				break;
			}
		}
	}
}

static inline int getSerializableDeclaredMembersNum(hkArrayBase<hkDataClass::MemberInfo>& membersInfo)
{
	int count = membersInfo.getSize();
	for( int i = 0; i < membersInfo.getSize(); ++i )
	{
		if( membersInfo[i].m_type->isVoid())
		{
			--count;
		}
	}
	return count;
}

static hkBool32 equalOriginalClass(const hkDataClass& original, const hkDataClass& calculated, hkStringMap<int>& doneClasses)
{
	int done = doneClasses.getWithDefault(original.getName(), -1);
	if( done != -1 )
	{
		return hkBool32(done);
	}
	if( !((original.getParent().isNull() && calculated.getParent().isNull())
			|| (!original.getParent().isNull()
				&& !calculated.getParent().isNull()
				&& hkString::strCmp(original.getParent().getName(), calculated.getParent().getName()) == 0)) )
	{
		doneClasses.insert(original.getName(), false);
		return false;
	}
	if( !original.getParent().isNull() && !equalOriginalClass(original.getParent(), calculated.getParent(), doneClasses))
	{
		return false;
	}
	// compare members
	{
		hkArray<hkDataClass::MemberInfo>::Temp originalMembersInfo(original.getNumDeclaredMembers());
		original.getAllDeclaredMemberInfo(originalMembersInfo);
		hkArray<hkDataClass::MemberInfo>::Temp calculatedMembersInfo(calculated.getNumDeclaredMembers());
		calculated.getAllDeclaredMemberInfo(calculatedMembersInfo);
		if( getSerializableDeclaredMembersNum(originalMembersInfo) != getSerializableDeclaredMembersNum(calculatedMembersInfo) )
		{
			doneClasses.insert(original.getName(), false);
			return false;
		}
		for( int i = 0; i < originalMembersInfo.getSize(); ++i )
		{
			hkDataClass::MemberInfo& originalMember = originalMembersInfo[i];
			if( originalMember.m_type->isVoid())
			{
				continue;
			}
			int calculatedMemberIdx = calculated.getDeclaredMemberIndexByName(originalMember.m_name);
			if( calculatedMemberIdx == -1 )
			{
				doneClasses.insert(original.getName(), false);
				return false;
			}
			hkDataClass::MemberInfo calculatedMember;
			calculated.getDeclaredMemberInfo(calculatedMemberIdx, calculatedMember);

			if( !originalMember.m_type->isEqual( calculatedMember.m_type))
			{
				doneClasses.insert(original.getName(), false);
				return false;
			}
			hkTypeManager::Type* originalTerm = originalMember.m_type->findTerminal();
			hkTypeManager::Type* calculatedTerm = calculatedMember.m_type->findTerminal();

			if (originalTerm->isClass())
			{
				// Need to find the class
				hkDataClass origClass = original.getWorld()->findClass(originalTerm->getTypeName());
				hkDataClass calcClass = calculated.getWorld()->findClass(calculatedTerm->getTypeName());

				if( !origClass.isNull() && !calcClass.isNull() && !calcClass.isSuperClass(origClass) )
				{
					doneClasses.insert(original.getName(), false);
					return false;
				}
			}
		}
	}

	doneClasses.insert(original.getName(), true);
	return true;
}

hkResult hkVersionPatchManager::preparePatches(hkDataWorld& worldToUpdate, ClassWrapper* wrapper, hkArray<const hkVersionPatchManager::PatchInfo*>& patchInfosOut) const
{
	// 1: Get all classes data from the world that needs to be updated
	hkArray<hkDataClassImpl*>::Temp originalClasses;
	worldToUpdate.findAllClasses(originalClasses);

	// 2: Transform all classes data into class version infos
	hkArray<ClassVersion> classes;
	classes.reserveExactly(originalClasses.getSize());
	for( int i = 0; i < originalClasses.getSize(); ++i )
	{
		const hkDataClassImpl* c = originalClasses[i];
		if( c->getVersion() >= 0 )
		{
			ClassVersion info = {c->getName(), c->getVersion()};
			classes.pushBackUnchecked(info);
		}
		else
		{
			#if HAVOK_BUILD_NUMBER == 0 // internal build, we allow some sloppiness
				HK_WARN(0x34a667ca, "Loading class " << c->getName() << " which is marked as under development (negative version number). It will bypass versioning and may not load correctly");
			#else // release build
				HK_ERROR(0x54d3b666, "Intermediate version found in a release build. The asset probably needs to be re-exported");
			#endif
		}
	}

	// 3: Look for class creation patches in the full patch list and store them in idxNewSrcFromUid (new class/version -> patch index)
	hkDataWorldDict localWorld;
	hkPointerMap<hkUint64, hkInt32> idxNewSrcFromUid;
	for( int i = 0; i < m_patchInfos.getSize(); ++i )
	{
		const PatchInfo* p = m_patchInfos[i];
		if( !isValidPatch(p) )
		{
			return HK_FAILURE;
		}
		if( p->oldVersion == -1/*HK_CLASS_ADDED*/ )
		{
			hkUint64 uid = m_uidFromClassVersion->get(p->newName, p->newVersion);
			if( idxNewSrcFromUid.hasKey(uid) == false )
			{
				idxNewSrcFromUid.insert( uid, i );
				TRACE(("'HK_CLASS_ADDED'\t%s(%d), index = %d\n", p->newName, p->newVersion, i));
			}
			else
			{
				HK_ASSERT3(0x24f1e017, false, "Found duplicated patch for new class '" << p->newName << "' (version " << p->newVersion << ").\n"\
					"Make sure you register the new class patch once with hkVersionPatchManager.");
				return HK_FAILURE;
			}
		}
	}

	TRACE(("\nFind required patches only...\n"));
	hkPointerMap<hkUint64, hkInt32> patchesDone;
	patchInfosOut.clear();
	{
		// 4: For each class that potentially needs to be updated, we look and find a patch having
		// as source the class, but we also save in todo all the patches that are dependencies to
		// that patch. todo is ordered using the patch index, so we don't loose the dependency
		// order.
		HeapArray<UidItem> todo;
		for( int versionIndex = 0; versionIndex < classes.getSize(); ++versionIndex )
		{
			const ClassVersion& ver = classes[versionIndex];
			setAllDependeciesToDo(todo, ver.name, ver.version, worldToUpdate, *this, idxNewSrcFromUid, patchesDone);
		}

		// 5: For each patch saved in todo (starting from the last one), we push it at the end of patchInfosOut
		// and look for a patch having as source the patched class (also pushing in todo the dependencies for
		// the new patch).
 		while( todo.getSize() )
 		{
			const UidItem top = todo.top(); // copy
			todo.popTop();
			if( top.patchIndex != -1 && m_patchInfos[top.patchIndex]->oldVersion == -1/*HK_CLASS_ADDED*/ )
			{
				HK_ASSERT(0x21d618c5, patchesDone.getWithDefault(top.uid, 0) != 2);
				patchesDone.insert(top.uid, 1);
			}
			else
			{
				patchesDone.insert(top.uid, 2);
			}
 			if( top.patchIndex >= 0 )
			{
				patchInfosOut.pushBack( getPatch(top.patchIndex) );
				const hkVersionPatchManager::PatchInfo* pinfo = getPatch(top.patchIndex);
 				const char* newName = pinfo->newName ? pinfo->newName : pinfo->oldName;
				setAllDependeciesToDo(todo, newName, pinfo->newVersion, worldToUpdate, *this, idxNewSrcFromUid, patchesDone);
 			}
 			else // no patch for uid. either it is the latest version or has been removed
 			{
 				const char* name = getClassName(top.uid);
 				int version = getClassVersion(top.uid);
 				if( version == -1 )
 				{
 					 // was removed
 				}
  				else
 				{
					hkDataClass klass = wrapper->wrapClass(&localWorld, name);

					if (klass.getImplementation() == HK_NULL)
					{
						HK_WARN_ALWAYS(0x3f79ddb0, "Class " << name << " is not registered. If this is a Havok class, make sure the class's product reflection is enabled near where hkProductFeatures.cxx is included. Otherwise, check your own class registration.");
						return HK_FAILURE;
					}
					if( klass.getVersion() != version )
					{
						HK_WARN_ALWAYS(0x3f79ddb1, "Source contains " << name << " version " << version
							<< ", but  " << klass.getVersion() << " is the current version.\n"
							<< "Make sure required patches are registered to update this class." );
						return HK_FAILURE;
					}
				}
 			}
 		}
	}
	TRACE(("Done.\n"));

	//
	// 6: Reverse order - reversed actions
	//
	for( int patchIndex = patchInfosOut.getSize()-1; patchIndex >= 0; --patchIndex )
	{
		const hkVersionPatchManager::PatchInfo* pinfo = patchInfosOut[patchIndex];
		const char* oldClassName = pinfo->oldName;
		const char* newClassName = pinfo->newName ? pinfo->newName : pinfo->oldName;
		hkDataClassImpl* klassImpl = localWorld.findClass( newClassName );
		hkDataClass klass( klassImpl );
		if( klass.isNull() )
		{
			// reverse class removal
			HK_ASSERT2(0x1a158d75, pinfo->newVersion == -1/*HK_CLASS_REMOVED*/,
				"Patches are out of order, probably because of missing dependencies\n.Check that the order of the local variable patchInfos is correct.");
			hkDataClass::Cinfo cinfo;
			cinfo.name = newClassName;
			cinfo.version = pinfo->oldVersion;
			cinfo.parent = HK_NULL;
			klass = localWorld.newClass(cinfo);
			HK_ASSERT(0x415ee5a6, localWorld.findClass( cinfo.name ));
			//TRACE(("+\tclass '%s' ver %d\n", cinfo.name, cinfo.version));
		}
#ifdef HK_DEBUG
		const ClassVersion tmp = {newClassName, pinfo->newVersion};
		HK_ASSERT3(0x7ea4bebf, classes.indexOf(tmp) == -1, "hkVersionPatchManager is trying to patch " << newClassName << " to version " << pinfo->newVersion << " but it is already in the world. Either the loaded asset or the patches are inconsistent." );
#endif
		TRACE(("LOOK %3i %s:0x%p -> %s:0x%p\n", patchIndex, (oldClassName ? oldClassName : newClassName), (void*)(hkUlong)pinfo->oldVersion, newClassName, (void*)(hkUlong)pinfo->newVersion));

		for( int componentIndex = pinfo->numComponent - 1; componentIndex >= 0 ; --componentIndex )
		{
			const hkVersionPatchManager::PatchInfo::Component& pc = pinfo->component[componentIndex];
			switch( pc.type )
			{
				case hkVersionPatchManager::PATCH_MEMBER_RENAMED:
				{
					const hkVersionPatchManager::MemberRenamedPatch* mem = reinterpret_cast<const hkVersionPatchManager::MemberRenamedPatch*>(pc.patch);
					localWorld.renameClassMember(klass, mem->newName, mem->oldName);
					break;
				}
				case hkVersionPatchManager::PATCH_MEMBER_REMOVED:
				{
					const hkVersionPatchManager::MemberRemovedPatch* mem = reinterpret_cast<const hkVersionPatchManager::MemberRemovedPatch*>(pc.patch);
					hkDataObject::Type type = localWorld.getTypeManager().getType(mem->type, mem->typeName, mem->tuples);

					localWorld.addClassMember(klass, mem->name, type, HK_NULL);
					break;
				}
				case hkVersionPatchManager::PATCH_MEMBER_ADDED:
				{
					const hkVersionPatchManager::MemberAddedPatch* mem = reinterpret_cast<const hkVersionPatchManager::MemberAddedPatch*>(pc.patch);
					localWorld.removeClassMember(klass, mem->name);
					break;
				}
				case hkVersionPatchManager::PATCH_FUNCTION:
				{
					break;
				}

				case hkVersionPatchManager::PATCH_MEMBER_DEFAULT_SET:
				{
					break;
				}
				case hkVersionPatchManager::PATCH_DEPENDS:
				{
					const hkVersionPatchManager::DependsPatch* dp = reinterpret_cast<const hkVersionPatchManager::DependsPatch*>(pc.patch);
					hkDataClass k( localWorld.findClass( dp->name ) );
					HK_ASSERT2(0x415ee5a6, !k.isNull(), dp->name);
					HK_ASSERT(0x2eaa1fbb, k.getVersion() == dp->version );
					break;
				}
				case hkVersionPatchManager::PATCH_CAST:
				{
					break;
				}
				case hkVersionPatchManager::PATCH_PARENT_SET:
				{
					const hkVersionPatchManager::SetParentPatch* spp = reinterpret_cast<const hkVersionPatchManager::SetParentPatch*>(pc.patch);
					if( spp->oldParent )
					{
						hkDataClass oldP( localWorld.findClass( spp->oldParent ) );
						localWorld.setClassParent(klass, oldP);
					}
					else
					{
						hkDataClass oldP( HK_NULL );
						localWorld.setClassParent(klass, oldP);
					}
					break;
				}
				default:
				{
					HK_ASSERT(0x5d5f8550, 0);
				}
			}
		}
		if( pinfo->oldVersion == -1/*HK_CLASS_ADDED*/ )
		{
			HK_ASSERT(0x1458d69b, !oldClassName);
			// reverse class added
			localWorld.removeClass(klass);
			HK_ASSERT(0x243646e8, localWorld.findClass( newClassName ) == HK_NULL );
		}
		else
		{
			if( (newClassName != oldClassName) && hkString::strCmp(newClassName,oldClassName) )
			{
				// reverse class renamed
				HK_ASSERT(0x223fedda, oldClassName);
				HK_ASSERT(0x415ee5a7, localWorld.findClass( oldClassName ) == HK_NULL );
				localWorld.renameClass(klass, oldClassName);
				HK_ASSERT(0x415ee5a8, localWorld.findClass( newClassName ) == HK_NULL );
			}
			localWorld.setClassVersion(klass, pinfo->oldVersion);
		}
	}

	//
	// 7: compare classes
	//
	{
		hkArray<hkDataClassImpl*>::Temp calculatedClasses;
		localWorld.findAllClasses(calculatedClasses);

		hkStringMap<hkDataClassImpl*> calcClassFromName;
		hkStringMap<int> doneClasses;
		for( int i = 0; i < calculatedClasses.getSize(); ++i )
		{
			calcClassFromName.insert(calculatedClasses[i]->getName(), calculatedClasses[i]);
		}
		for( int i = 0; i < originalClasses.getSize(); ++i )
		{
			hkDataClassImpl* origClass = originalClasses[i];
			hkDataClassImpl* calcClass = calcClassFromName.getWithDefault(origClass->getName(), HK_NULL);
			if( !calcClass )
			{
				if( origClass->getVersion() >= 0 )
				{
					HK_WARN(0x39edb60c, "Class " << origClass->getName() << " is not found in the reverse-engineered class list.");
				}
				continue;
			}
			HK_ON_DEBUG(if( !equalOriginalClass(origClass, calcClass, doneClasses) ))
			{
				HK_WARN(0x265cc0f9, "The class " << origClass->getName() << " when run through the patching system does not match the currrent class. The most likely cause is changes to the current class without a corresponding patch");
			}
			calcClassFromName.remove(origClass->getName());
		}
		int addedClasses = 0;
		for( hkStringMap<hkDataClassImpl*>::Iterator it = calcClassFromName.getIterator(); calcClassFromName.isValid(it); it = calcClassFromName.getNext(it) )
		{
			//HK_WARN(0x58b3f015, "Class " << calcClassFromName.getValue(it)->getName() << " (" << calcClassFromName.getValue(it)->getVersion() << ") is missing in the provided world. Adding it.");
			worldToUpdate.copyClassFromWorld(calcClassFromName.getValue(it)->getName(), localWorld);
			addedClasses += 1;
		}

		if( (originalClasses.getSize() + addedClasses) != calculatedClasses.getSize() )
		{
			HK_WARN(0x13821fce, "The number of reverse-engineered classes is different from number of original classes. This may because of missing patches, intermediate versions or missing patch dependencies.");
		}
	}
	return HK_SUCCESS;
}

static hkResult applyPatchesFinally(hkDataWorld& worldToUpdate, const hkArray<const hkVersionPatchManager::PatchInfo*>& patchInfos);

hkResult hkVersionPatchManager::applyPatches(hkDataWorld& worldToUpdate, ClassWrapper* classWrapper) const
{
	hkDefaultClassWrapper defWrapper;
	if (classWrapper == HK_NULL)
	{
		classWrapper = &defWrapper;
	}

	HK_ASSERT2( 0x30173b35, m_patchIndexFromUid.getSize() != 0 || m_patchInfos.getSize() == 0, "Patches exist, but patch dependencies are not computed. Did you call recomputePatchDependencies()?");
	hkArray<const hkVersionPatchManager::PatchInfo*> patchInfos;

	// select patches and build patch dependencies using loaded classes info
	hkResult res = preparePatches(worldToUpdate, classWrapper, patchInfos);
	if( res == HK_SUCCESS )
	{
		res = applyPatchesFinally(worldToUpdate, patchInfos);
	}

	return res;
}

hkResult hkVersionPatchManager::applyPatchesDebug(hkDataWorld& worldToUpdate) const
{
	HK_ASSERT2( 0x30173b36, m_patchIndexFromUid.getSize() != 0 || m_patchInfos.getSize() == 0, "Patches exist, but patch dependencies are not computed. Did you call recomputePatchDependencies()?");
	hkResult res = applyPatchesFinally(worldToUpdate, m_patchInfos);
	return res;
}

static hkResult applyPatchesFinally(hkDataWorld& worldToUpdate, const hkArray<const hkVersionPatchManager::PatchInfo*>& patchInfos)
{
	for( int patchIndex = 0; patchIndex < patchInfos.getSize(); ++patchIndex )
	{
		const hkVersionPatchManager::PatchInfo* pinfo = patchInfos[patchIndex];
		HK_ASSERT(0x33fa6b07, pinfo);

#if defined(HK_DEBUG)
		for( int componentIndex = 0; componentIndex < pinfo->numComponent; ++componentIndex )
		{
			const hkVersionPatchManager::PatchInfo::Component& pc = pinfo->component[componentIndex];
			switch( pc.type )
			{
				case hkVersionPatchManager::PATCH_DEPENDS:
				{
					const hkVersionPatchManager::DependsPatch* dpatch = static_cast<const hkVersionPatchManager::DependsPatch*>(pc.patch);
					HK_ASSERT(0x101762d2, dpatch);
					HK_ASSERT(0x4b25d1a1, dpatch->name && dpatch->version != -1);
					hkDataClassImpl* dklass = worldToUpdate.findClass(dpatch->name);
					if( dklass == HK_NULL )
					{
						hkStringBuf text;
						text.printf("Could not find patch dependency %s v%d when updating %s v%d to %s v%d", dpatch->name,
							dpatch->version, pinfo->oldName, pinfo->oldVersion, pinfo->newName, pinfo->newVersion);
						HK_TRACE(text.cString());
						HK_ASSERT2(0x217729aa, false, "Patches are out of order or invalid. Did you call hkVersionPatchManager::recomputePatchDependencies()?");
						return HK_FAILURE;
					}
					break;
				}
				default:
					// do nothing
					break;
			}
		}
#endif

		const char* className = pinfo->oldName ? pinfo->oldName : pinfo->newName;
		if( pinfo->oldVersion == -1 /*HK_CLASS_ADDED*/)
		{
			hkDataClass::Cinfo cinfo;
			cinfo.name = className;
			cinfo.version = pinfo->newVersion;
			cinfo.parent = HK_NULL;
			if( worldToUpdate.findClass(className) != HK_NULL )
			{
				HK_ASSERT3(0x437782db, false, "Unable to apply 'HK_CLASS_ADDED' patch for class '" << className << "', version " << (void*)(hkUlong)cinfo.version << ".\n"\
					"The class '" << className << "' exists in the provided world.\n"\
					"You probably should call hkVersionPatchManager::applyPatches() with 'usePatchesForWorldClassesOnly' set to true.");
				return HK_FAILURE;
			}
			worldToUpdate.newClass(cinfo);
		}
		HK_ASSERT3(0x437582db, worldToUpdate.findClass(className) != HK_NULL, "Cannot find class " << className );
		hkDataClass klass( worldToUpdate.findClass( className ) );
		TRACE((">>\t%s, %d\t(%d)%s\n", className, klass.getVersion(), patchIndex, (pinfo->oldVersion == -1 ? " +" : "")));

		if( pinfo->oldName && pinfo->newName && hkString::strCmp(pinfo->oldName, pinfo->newName) != 0 )
		{
			worldToUpdate.renameClass(klass, pinfo->newName);
			// Need to update the class name or patch functions will never
			// be called for the new, renamed class
			className = pinfo->newName;
		}
		for( int componentIndex = 0; componentIndex < pinfo->numComponent; ++componentIndex )
		{
			const hkVersionPatchManager::PatchInfo::Component& pc = pinfo->component[componentIndex];

			switch( pc.type )
			{
				case hkVersionPatchManager::PATCH_MEMBER_RENAMED:
				{
					const hkVersionPatchManager::MemberRenamedPatch* mem = reinterpret_cast<const hkVersionPatchManager::MemberRenamedPatch*>(pc.patch);
					worldToUpdate.renameClassMember(klass, mem->oldName, mem->newName);
					break;
				}
				case hkVersionPatchManager::PATCH_MEMBER_ADDED:
				{
					const hkVersionPatchManager::MemberAddedPatch* mem = reinterpret_cast<const hkVersionPatchManager::MemberAddedPatch*>(pc.patch);
					hkDataObject::Type type = worldToUpdate.getTypeManager().getType(mem->type, mem->typeName, mem->tuples);
					worldToUpdate.addClassMember(klass, mem->name, type, mem->defaultPtr);
					break;
				}
				case hkVersionPatchManager::PATCH_MEMBER_REMOVED:
				{
					const hkVersionPatchManager::MemberRemovedPatch* mem = reinterpret_cast<const hkVersionPatchManager::MemberRemovedPatch*>(pc.patch);
					worldToUpdate.removeClassMember(klass, mem->name);
					break;
				}
				case hkVersionPatchManager::PATCH_MEMBER_DEFAULT_SET:
				{
					const hkVersionPatchManager::DefaultChangedPatch* mem = reinterpret_cast<const hkVersionPatchManager::DefaultChangedPatch*>(pc.patch);
					worldToUpdate.setClassMemberDefault(klass, mem->name, mem->defaultPtr);
					break;
				}

				case hkVersionPatchManager::PATCH_FUNCTION:
				{
 					const hkVersionPatchManager::Function func = reinterpret_cast<const hkVersionPatchManager::FunctionPatch*>(pc.patch)->function;
					hkArray<hkDataObjectImpl*>::Temp objs;
					worldToUpdate.findObjectsByBaseClass(className, objs);
					for( int index = 0; index < objs.getSize(); ++index )
					{
						hkDataObject obj( objs[index] );
 						(*func)(obj);
 					}
					break;
				}
				case hkVersionPatchManager::PATCH_PARENT_SET:
				{
					//todo.sk what happens when members are orphaned?
					const hkVersionPatchManager::SetParentPatch* spp = reinterpret_cast<const hkVersionPatchManager::SetParentPatch*>(pc.patch);
					HK_ON_DEBUG(if( spp->newParent ))
					{
						HK_ASSERT(0x4dcf113b, worldToUpdate.findClass(spp->newParent));
					}
					hkDataClass newP( spp->newParent ? worldToUpdate.findClass( spp->newParent ) : HK_NULL );
					worldToUpdate.setClassParent(klass, newP);
					break;
				}
				case hkVersionPatchManager::PATCH_DEPENDS:
				{
					break;
				}
				case hkVersionPatchManager::PATCH_CAST:
				{
					// remap objects' class to class (cp->name)
					const hkVersionPatchManager::CastPatch* cp = reinterpret_cast<const hkVersionPatchManager::CastPatch*>(pc.patch);
					hkDataClass castClass( worldToUpdate.findClass( cp->name ) );
					hkArray<hkDataObjectImpl*>::Temp objs;
					worldToUpdate.findObjectsByExactClass(className, objs);
					for( int index = 0; index < objs.getSize(); ++index )
					{
						hkDataObject obj( objs[index] );
						//if( !castClass.isSuperClass(obj.getClass()) )
						{
							worldToUpdate.castObject(obj, castClass);
						}
					}
					break;
				}
				default:
				{
					HK_ASSERT(0x7d3a3d97, 0);
				}
			}
		}
		if( pinfo->newVersion == -1 /*HK_CLASS_REMOVED*/)
		{
			worldToUpdate.removeClass(klass);
		}
		else if( pinfo->oldVersion != -1 /*HK_CLASS_ADDED*/ && pinfo->oldVersion != pinfo->newVersion )
		{
			worldToUpdate.setClassVersion(klass, pinfo->newVersion);
		}
	}

	return HK_SUCCESS;
}

void hkVersionPatchManager::addPatch(const PatchInfo* p)
{
	if( !isValidPatch(p) )
	{
		return;
	}
	m_patchInfos.pushBack(p);
	m_patchIndexFromUid.clear();
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
