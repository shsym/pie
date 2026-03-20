import { describe, it, expect, vi, beforeEach } from 'vitest';
import { KvPageManager } from '../kv-page-manager.js';

// Mock the Queue's newKvPages method
const mockNewKvPages = vi.fn();
const mockQueue = {
    newKvPages: mockNewKvPages,
} as any;

describe('KvPageManager', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should start with zero pages', () => {
        const manager = new KvPageManager(mockQueue, 128);
        expect(manager.pageCount).toBe(0);
        expect(manager.lastPageLen).toBe(0);
        expect(manager.totalTokens).toBe(0);
    });

    it('should allocate pages on grow', () => {
        const mockPage = { ptr: 1, ref: vi.fn(), release: vi.fn() };
        mockNewKvPages.mockReturnValue([mockPage]);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(100);

        expect(mockNewKvPages).toHaveBeenCalledWith(1);
        expect(manager.pageCount).toBe(1);
        expect(manager.lastPageLen).toBe(100);
    });

    it('should allocate multiple pages for large grows', () => {
        const mockPages = [
            { ptr: 1, ref: vi.fn(), release: vi.fn() },
            { ptr: 2, ref: vi.fn(), release: vi.fn() },
        ];
        mockNewKvPages.mockReturnValue(mockPages);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(200);

        expect(mockNewKvPages).toHaveBeenCalledWith(2);
        expect(manager.pageCount).toBe(2);
        expect(manager.lastPageLen).toBe(72); // 200 % 128 = 72
    });

    it('should shrink pages when tokens are removed', () => {
        const mockPages = [
            { ptr: 1, ref: vi.fn(), release: vi.fn() },
            { ptr: 2, ref: vi.fn(), release: vi.fn() },
        ];
        mockNewKvPages.mockReturnValue(mockPages);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(200); // 2 pages, lastPageLen = 72

        expect(manager.pageCount).toBe(2);
        expect(manager.totalTokens).toBe(200);

        // Shrink by 100 tokens -> 100 tokens remaining -> 1 page needed
        manager.shrink(100);

        expect(manager.pageCount).toBe(1);
        expect(manager.totalTokens).toBe(100);
        expect(manager.lastPageLen).toBe(100);
        // Second page should have been released
        expect(mockPages[1].release).toHaveBeenCalled();
    });

    it('should handle shrink to zero', () => {
        const mockPage = { ptr: 1, ref: vi.fn(), release: vi.fn() };
        mockNewKvPages.mockReturnValue([mockPage]);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(50);

        expect(manager.pageCount).toBe(1);

        manager.shrink(50);

        expect(manager.pageCount).toBe(0);
        expect(manager.totalTokens).toBe(0);
        expect(mockPage.release).toHaveBeenCalled();
    });

    it('should throw on underflow', () => {
        const manager = new KvPageManager(mockQueue, 128);

        expect(() => manager.shrink(10)).toThrow('underflow');
    });

    it('should fork and share only full pages', () => {
        const mockPages = [
            { ptr: 1, ref: vi.fn(), release: vi.fn() },
            { ptr: 2, ref: vi.fn(), release: vi.fn() },
        ];
        mockNewKvPages.mockReturnValue(mockPages);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(200); // 2 pages: first full (128), second partial (72)

        const { manager: forked, droppedTokenCount } = manager.fork();

        // Fork should only keep full pages (1 page with 128 tokens)
        expect(forked.pageCount).toBe(1);
        expect(forked.totalTokens).toBe(128);
        expect(droppedTokenCount).toBe(72); // 200 - 128

        // First page should have ref() called (shared)
        expect(mockPages[0].ref).toHaveBeenCalled();
        // Second page should NOT have ref() called (not shared)
        expect(mockPages[1].ref).not.toHaveBeenCalled();
    });

    it('should fork with no pages when only partial page exists', () => {
        const mockPage = { ptr: 1, ref: vi.fn(), release: vi.fn() };
        mockNewKvPages.mockReturnValue([mockPage]);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(50); // 1 partial page

        const { manager: forked, droppedTokenCount } = manager.fork();

        // No full pages to share
        expect(forked.pageCount).toBe(0);
        expect(forked.totalTokens).toBe(0);
        expect(droppedTokenCount).toBe(50);
    });

    it('should release all pages', () => {
        const mockPages = [
            { ptr: 1, ref: vi.fn(), release: vi.fn() },
            { ptr: 2, ref: vi.fn(), release: vi.fn() },
        ];
        mockNewKvPages.mockReturnValue(mockPages);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(200);

        manager.release();

        expect(mockPages[0].release).toHaveBeenCalled();
        expect(mockPages[1].release).toHaveBeenCalled();
        expect(manager.pageCount).toBe(0);
        expect(manager.totalTokens).toBe(0);
    });

    it('should handle multiple release calls safely', () => {
        const mockPage = { ptr: 1, ref: vi.fn(), release: vi.fn() };
        mockNewKvPages.mockReturnValue([mockPage]);

        const manager = new KvPageManager(mockQueue, 128);
        manager.grow(50);

        manager.release();
        manager.release(); // Should not throw

        expect(mockPage.release).toHaveBeenCalledTimes(1);
    });

    it('should adopt pages from another manager', () => {
        const mockPages1 = [
            { ptr: 1, ref: vi.fn(), release: vi.fn() },
        ];
        const mockPages2 = [
            { ptr: 2, ref: vi.fn(), release: vi.fn() },
            { ptr: 3, ref: vi.fn(), release: vi.fn() },
        ];

        mockNewKvPages
            .mockReturnValueOnce(mockPages1)
            .mockReturnValueOnce(mockPages2);

        const manager1 = new KvPageManager(mockQueue, 128);
        manager1.grow(50);

        const manager2 = new KvPageManager(mockQueue, 128);
        manager2.grow(200);

        // Adopt pages from manager2
        manager1.adopt(manager2);

        // Manager1 should have released its old page
        expect(mockPages1[0].release).toHaveBeenCalled();

        // Manager1 should now have manager2's pages with incremented refs
        expect(manager1.pageCount).toBe(2);
        expect(mockPages2[0].ref).toHaveBeenCalled();
        expect(mockPages2[1].ref).toHaveBeenCalled();
    });

    it('should import pages from external source', () => {
        const externalPages = [
            { ptr: 10, ref: vi.fn(), release: vi.fn() },
            { ptr: 11, ref: vi.fn(), release: vi.fn() },
        ] as any;

        const manager = new KvPageManager(mockQueue, 128);
        manager.importPages(externalPages, 64);

        expect(manager.pageCount).toBe(2);
        expect(manager.lastPageLen).toBe(64);
        expect(manager.totalTokens).toBe(128 + 64); // 192
        expect(manager.ptrs).toEqual([10, 11]);
    });
});